import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ROTVisionTransformer(nn.Module):
    """
    ROT-ViT:
      1) x проходит через embeddings и первые L-1 transformer blocks;
      2) получаем z^{L-1};
      3) branch 1: z^{L-1} -> last block -> encoder_norm -> head -> l_1;
      4) branch 2: rotate(z^{L-1}) -> last block -> encoder_norm -> head -> l_2.

    Важно:
      - CLS-token не поворачивается;
      - поворачиваются только patch-токены, которые reshape-ятся в 2D сетку;
      - последний transformer block и classifier head общие для обеих веток.
    """

    def __init__(
        self,
        base_vit,
        max_rotation_degrees=None,
        rotate_in_eval=False,
    ):
        super().__init__()

        self.base_vit = base_vit
        self.max_rotation_degrees = max_rotation_degrees
        self.rotate_in_eval = rotate_in_eval
        self.vis = False

    def forward(self, x):
        """
        Возвращает:
            logits_per_branch:   [2, B, num_classes]
            features_per_branch: [2, B, 197, hidden_size]
            angles:              [B]
        """

        hidden_states = self.base_vit.embeddings(x)

        for layer_block in self.base_vit.encoder.layer[:-1]:
            hidden_states, _ = layer_block(hidden_states)

        z_l_minus_1 = hidden_states
        z_l_minus_1_sec_branch = hidden_states
        last_block = self.base_vit.encoder.layer[-1]

        # branch 1
        z_1_l, _ = last_block(z_l_minus_1)
        z_1_l = self.base_vit.encoder.encoder_norm(z_1_l)
        logits_1 = self.base_vit.head(z_1_l[:, 0])

        # Branch 2
        if self.training:
            z_rot, angles = self.rotate_features_random(z_l_minus_1_sec_branch)
        else:
            z_rot = z_l_minus_1_sec_branch
            # поворачиваем на нулевой угол
            # тут должны для всего батч сайза так сделать
            angles = torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=z_l_minus_1_sec_branch.dtype,
            )

        z_2_l, _ = last_block(z_rot)
        z_2_l = self.base_vit.encoder.encoder_norm(z_2_l)
        logits_2 = self.base_vit.head(z_2_l[:, 0])

        logits_per_branch = torch.stack([logits_1, logits_2], dim=0)
        features_per_branch = torch.stack([z_1_l, z_2_l], dim=0)

        # return logits_per_branch, features_per_branch

        return {
            "logits_per_branch": logits_per_branch,
            "features_per_branch": features_per_branch,
            "angles": angles,
            "logits_1": logits_1,
            "logits_2": logits_2,
            "features_1": z_1_l,
            "features_2": z_2_l,
        }

    def rotate_features_random(self, features):
        """
        features: [B, 197, C]

        Поворачиваем только patch-токены:
            [B, 196, C] -> [B, C, 14, 14] -> grid_sample -> [B, 196, C]

        CLS-token остается без изменений.
        """

        batch_size, _, hidden_size = features.shape

        cls_token = features[:, :1, :]
        patch_tokens = features[:, 1:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            raise ValueError(f"Number of patch tokens must be a square, got {num_patches}.")

        # [B, 196, C] -> [B, C, 14, 14]
        patch_map = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_size,
            grid_size,
            grid_size,
        )

        if self.max_rotation_degrees == 0:
            angles = torch.zeros(
                batch_size,
                device=features.device,
                dtype=features.dtype,
            )
            return features, angles

        angles = torch.empty(batch_size, device=features.device, dtype=features.dtype).uniform_(
            -self.max_rotation_degrees, self.max_rotation_degrees
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

        # [B, C, 14, 14] -> [B, 196, C]
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
