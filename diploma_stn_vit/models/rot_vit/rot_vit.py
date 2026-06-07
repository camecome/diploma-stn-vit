import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROTVisionTransformer(nn.Module):
    """
    ROT-ViT:
      1) x проходит через embeddings и первые L-1 transformer blocks;
      2) получаем z^{L-1};
      3) branch 1: z^{L-1} -> last block 1 -> encoder_norm 1 -> head 1 -> l_1;
      4) branch 2: rotate(z^{L-1}) -> last block 2 -> encoder_norm 2 -> head 2 -> l_2.

    Важно:
      - CLS-token не поворачивается;
      - поворачиваются только patch-токены, которые reshape-ятся в 2D сетку;
      - последний transformer block, norm и classifier head физически разные для двух веток;
      - при инициализации параметры второй ветки копируются из base_vit.
    """

    FIRST_BRANCH = 0
    SECOND_BRANCH = 1

    def __init__(
        self,
        base_vit,
        max_rotation_degrees=None,
        rotate_in_eval=False,
    ):
        super().__init__()

        self.max_rotation_degrees = max_rotation_degrees
        self.rotate_in_eval = rotate_in_eval
        self.vis = False

        self.common_embeddings = base_vit.transformer.embeddings
        self.common_layers = base_vit.transformer.encoder.layer[:-1]

        self.last_layers = nn.ModuleList(
            [
                copy.deepcopy(base_vit.transformer.encoder.layer[-1]),
                copy.deepcopy(base_vit.transformer.encoder.layer[-1]),
            ]
        )

        self.norms = nn.ModuleList(
            [
                copy.deepcopy(base_vit.transformer.encoder.encoder_norm),
                copy.deepcopy(base_vit.transformer.encoder.encoder_norm),
            ]
        )

        self.heads = nn.ModuleList(
            [
                copy.deepcopy(base_vit.head),
                copy.deepcopy(base_vit.head),
            ]
        )

    def forward(self, x):
        hidden_states = self.common_embeddings(x)

        for layer_block in self.common_layers:
            hidden_states, _ = layer_block(hidden_states)

        z_l_minus_1 = hidden_states

        # branch 1: без поворота
        logits_1, z_1_l = self.get_features_logits(
            branch_idx=self.FIRST_BRANCH,
            hidden_states=z_l_minus_1,
        )

        # branch 2: с поворотом в train, опционально с поворотом в eval
        if self.training or self.rotate_in_eval:
            z_rot, angles = self.rotate_features_random(z_l_minus_1)
        else:
            z_rot = z_l_minus_1
            angles = torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=z_l_minus_1.dtype,
            )

        logits_2, z_2_l = self.get_features_logits(
            branch_idx=self.SECOND_BRANCH,
            hidden_states=z_rot,
        )

        logits_per_branch = torch.stack([logits_1, logits_2], dim=0)
        features_per_branch = torch.stack([z_1_l, z_2_l], dim=0)

        return {
            "logits_per_branch": logits_per_branch,
            "features_per_branch": features_per_branch,
            "angles": angles,
            "logits_1": logits_1,
            "logits_2": logits_2,
            "features_1": z_1_l,
            "features_2": z_2_l,
        }

    def get_features_logits(self, branch_idx, hidden_states):
        hidden_states, _ = self.last_layers[branch_idx](hidden_states)
        hidden_states = self.norms[branch_idx](hidden_states)
        logits = self.heads[branch_idx](hidden_states[:, 0])

        return logits, hidden_states

    def rotate_features_random(self, features):
        batch_size, _, hidden_size = features.shape

        cls_token = features[:, :1, :]
        patch_tokens = features[:, 1:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            raise ValueError(f"Number of patch tokens must be a square, got {num_patches}.")

        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            grid_size,
            grid_size,
        )

        if self.max_rotation_degrees == 0:
            zero_angles = torch.zeros(
                batch_size,
                device=features.device,
                dtype=features.dtype,
            )
            return features, zero_angles

        angles = torch.empty(
            batch_size,
            device=features.device,
            dtype=features.dtype,
        ).uniform_(
            -self.max_rotation_degrees,
            self.max_rotation_degrees,
        )

        theta = self.angles_to_theta(angles)

        grid = F.affine_grid(
            theta,
            size=patch_map.size(),
            align_corners=False,
        )

        rotated_patch_map = F.grid_sample(
            patch_map,
            grid,
            padding_mode="zeros",
            align_corners=False,
        )

        rotated_patch_tokens = rotated_patch_map.reshape(
            batch_size,
            hidden_size,
            num_patches,
        ).transpose(1, 2)

        rotated_features = torch.cat([cls_token, rotated_patch_tokens], dim=1)

        return rotated_features, angles

    @staticmethod
    def angles_to_theta(angles_degrees):
        """
        angles_degrees: [B]

        Возвращает theta: [B, 2, 3] для affine_grid.
        """

        angles = angles_degrees * math.pi / 180.0

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        theta = torch.zeros(
            angles.shape[0],
            2,
            3,
            device=angles.device,
            dtype=angles.dtype,
        )

        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos

        return theta
