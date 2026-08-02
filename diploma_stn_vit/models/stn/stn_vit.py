import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

from .localization import ViTLocalization
from utils.augment_data_utils import get_safe_rotation_size


def rotate_batch(images: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got {images.shape}")

    if degrees.ndim != 1:
        raise ValueError(f"Expected degrees with shape [B], got {degrees.shape}")

    if images.shape[0] != degrees.shape[0]:
        raise ValueError(
            f"Batch size mismatch: images.shape[0]={images.shape[0]}, " f"degrees.shape[0]={degrees.shape[0]}"
        )

    batch_size = images.shape[0]

    radians = degrees * math.pi / 180.0
    cos = torch.cos(radians)
    sin = torch.sin(radians)

    theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)

    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos

    grid = F.affine_grid(theta, images.size(), align_corners=False)

    # обратное преобразование
    return F.grid_sample(
        images,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )


def rotate_images_without_black_borders(images, max_angle):
    batch_size, _, height, width = images.shape

    if height != width:
        raise ValueError(f"Expected square images, got height={height}, width={width}")

    original_size = height
    safe_size = get_safe_rotation_size(img_size=original_size, max_rotation_degrees=max_angle)

    if safe_size < original_size:
        raise ValueError(f"got invalid safe_size {safe_size}, original_size is {original_size}")

    resized_images = TF.resize(images, size=[safe_size, safe_size], antialias=True)
    degrees = torch.empty(batch_size, device=images.device).uniform_(-max_angle, max_angle)
    rotated_images = rotate_batch(resized_images, degrees)
    cropped_images = TF.center_crop(rotated_images, output_size=[original_size, original_size])

    # affine_grid задаёт обратное отображение координат, поэтому видимый поворот
    # изображения имеет знак, противоположный углу в affine-матрице.
    input_rotation_degrees = -degrees
    return cropped_images, input_rotation_degrees


def get_features_logits(layer, norm, head, hidden_states):
    # hidden_states [batch_size, 197, 768]

    hidden_states, _ = layer(hidden_states)
    hidden_states = norm(hidden_states)

    # [batch_size, 768]
    cls_token = hidden_states[:, 0]

    # [batch_size, 1k]
    logits = head(cls_token)

    # [batch_size, 1k], [batch_size, 197, 768]
    return logits, hidden_states


def make_training_batch(images, max_rotation_degrees, num_rotations):
    rotated_images = []
    input_rotation_degrees = []
    for _ in range(num_rotations):
        rotated_batch, rotation_degrees = rotate_images_without_black_borders(
            images=images,
            max_angle=max_rotation_degrees,
        )
        rotated_images.append(rotated_batch)
        input_rotation_degrees.append(rotation_degrees)

    # [self.num_branches * batch_size, C, H, W]
    training_batch = torch.cat([images, *rotated_images], dim=0)
    # [num_rotations, batch_size]
    input_rotation_degrees = torch.stack(input_rotation_degrees, dim=0)
    return training_batch, input_rotation_degrees


class SpatialTransformerViT(nn.Module):
    REFERENCE_BRANCH = 0
    STN_BRANCH = 1

    def __init__(
        self,
        base_vit,
        max_rotation_degrees,
        conv_channels=(512, 256, 32, 8),
        use_stn=True,
    ):
        super().__init__()

        self.max_rotation_degrees = max_rotation_degrees
        self.num_rotations = 1
        self.num_branches = self.num_rotations + 1
        self.use_stn = use_stn

        self.common_embeddings = base_vit.transformer.embeddings
        self.common_layers = base_vit.transformer.encoder.layer[:-1]

        self.last_layers = nn.ModuleList(
            [copy.deepcopy(base_vit.transformer.encoder.layer[-1]) for _ in range(self.num_branches)]
        )

        self.norms = nn.ModuleList(
            [copy.deepcopy(base_vit.transformer.encoder.encoder_norm) for _ in range(self.num_branches)]
        )

        self.heads = nn.ModuleList([copy.deepcopy(base_vit.head) for _ in range(self.num_branches)])

        # localization network предсказывает tan угла поворота
        self.loc_net = ViTLocalization(
            input_shape=[768, 14, 14],
            conv_channels=conv_channels,
        )

    def forward(self, images):
        if self.training:
            # logits, features, affine matrices, input rotation angles
            return self.forward_train(images)

        # [batch_size, 1k], [batch_size, 197, 768]
        return self.forward_eval(images, return_theta=True)

    def forward_train(self, images):
        batch_size = images.shape[0]

        # images [self.num_branches * batch_size, C, H, W]
        images, input_rotation_degrees = make_training_batch(
            images,
            self.max_rotation_degrees,
            self.num_rotations,
        )
        # hidden_states [self.num_branches * batch_size, 197, 768]
        hidden_states = self.forward_common_layers(images)

        logits_per_branch = []
        features_per_branch = []
        theta_per_branch = []
        predicted_angles_per_branch = []

        for branch_idx in range(self.num_branches):
            start = branch_idx * batch_size
            end = (branch_idx + 1) * batch_size
            # branch_hidden_states [batch_size, 197, 768]
            branch_hidden_states = hidden_states[start:end]
            a = None
            predicted_angles = None

            if branch_idx > self.REFERENCE_BRANCH:
                # a [batch_size, 2, 2]
                branch_hidden_states, a, predicted_angles = self.transform_patch_tokens(
                    branch_hidden_states,
                    return_theta=True,
                )

            logits, features = get_features_logits(
                layer=self.last_layers[branch_idx],
                norm=self.norms[branch_idx],
                head=self.heads[branch_idx],
                hidden_states=branch_hidden_states,
            )

            # logits [batch_size, 1k], features [batch_size, 197, 768]

            features_per_branch.append(features)
            logits_per_branch.append(logits)

            if a is not None:
                theta_per_branch.append(a)
                predicted_angles_per_branch.append(predicted_angles)

        # [self.num_branches, batch_size, 1k]
        logits_per_branch = torch.stack(logits_per_branch, dim=0)
        # [self.num_branches, batch_size, 197, 768]
        features_per_branch = torch.stack(features_per_branch, dim=0)
        # [batch_size, 2, 2]
        theta_per_branch = torch.stack(theta_per_branch, dim=0)
        # [num_rotations, batch_size]
        predicted_angles_per_branch = torch.stack(predicted_angles_per_branch, dim=0)

        return (
            logits_per_branch,
            features_per_branch,
            theta_per_branch,
            input_rotation_degrees,
            predicted_angles_per_branch,
        )

    def forward_eval(self, images, return_theta=False):
        if not self.use_stn:
            raise NotImplementedError("yet to implement eval logic for this")
        # [batch_size, 197, 768]
        hidden_states = self.forward_common_layers(images)

        # rotate hidden states
        # [batch_size, 197, 768]
        hidden_states, theta, predicted_angles = self.transform_patch_tokens(
            hidden_states,
            return_theta=return_theta,
        )

        logits, features = get_features_logits(
            layer=self.last_layers[self.STN_BRANCH],
            norm=self.norms[self.STN_BRANCH],
            head=self.heads[self.STN_BRANCH],
            hidden_states=hidden_states,
        )

        # [batch_size, 1k], [batch_size, 197, 768], [batch_size, 2, 2]
        return logits, features, theta, predicted_angles

    def forward_common_layers(self, images):
        hidden_states = self.common_embeddings(images)

        for layer in self.common_layers:
            hidden_states, _ = layer(hidden_states)

        return hidden_states

    def transform_patch_tokens(self, hidden_states_batch, return_theta=False):
        if not self.use_stn:
            if return_theta:
                return hidden_states_batch, None, None
            return hidden_states_batch

        # cls_token [batch_size, 1, 768]
        cls_token = hidden_states_batch[:, :1]

        # patch_tokens [batch_size, 196, 768]
        patch_tokens = hidden_states_batch[:, 1:]

        # patch_feature_map [batch_size, 768, 14, 14]
        patch_feature_map = self.tokens_to_feature_map(patch_tokens)

        # tan_theta [batch_size, 1]
        tan_theta = self.loc_net(patch_feature_map)

        # theta [batch_size, 2, 3], a [batch_size, 2, 2]
        theta, a, predicted_angles = self.build_affine_theta(tan_theta)

        # transformed_patch_feature_map [batch_size, 768, 14, 14]
        transformed_patch_feature_map = self.apply_stn(patch_feature_map, theta)

        # transformed_patch_tokens [batch_size, 196, 768]
        transformed_patch_tokens = self.feature_map_to_tokens(transformed_patch_feature_map)

        # [batch_size, 197, 768]
        transformed_hidden_states = torch.cat([cls_token, transformed_patch_tokens], dim=1)

        if return_theta:
            return transformed_hidden_states, a, predicted_angles

        return transformed_hidden_states

    def tokens_to_feature_map(self, patch_tokens):
        batch_size, num_patches, hidden_dim = patch_tokens.shape
        grid_size = int(num_patches**0.5)

        if grid_size * grid_size != num_patches:
            raise ValueError(f"num_patches must be a square number, got {num_patches}")

        return patch_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden_dim,
            grid_size,
            grid_size,
        )

    def feature_map_to_tokens(self, feature_map):
        return feature_map.flatten(2).transpose(1, 2)

    def apply_stn(self, feature_map, theta):
        if theta.ndim != 3 or theta.shape[1:] != (2, 3):
            raise ValueError(f"Expected theta with shape [B, 2, 3], got {theta.shape}")

        grid = F.affine_grid(
            theta,
            feature_map.size(),
            align_corners=False,
        )

        return F.grid_sample(
            feature_map,
            grid,
            padding_mode="zeros",
            align_corners=False,
        )

    def build_affine_theta(self, tan_theta):
        """
        tan_theta: [B, 1]

        Возвращает:
            theta: [B, 2, 3] — affine-матрица для affine_grid
            a:     [B, 2, 2] — левая квадратная подматрица A = R(theta)
            predicted_angles: [B] — предсказанный угол в градусах
        """
        if tan_theta.ndim != 2 or tan_theta.shape[1] != 1:
            raise ValueError(f"Expected tan_theta with shape [B, 1], got {tan_theta.shape}")

        batch_size = tan_theta.shape[0]

        # angle [B, 1]
        # torch.atan(tan_theta)
        angle = torch.atan2(tan_theta, torch.ones_like(tan_theta))
        predicted_angles = torch.rad2deg(angle.squeeze(1))

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # a: [B, 2, 2]
        a = torch.zeros(
            batch_size,
            2,
            2,
            device=tan_theta.device,
            dtype=tan_theta.dtype,
        )

        a[:, 0, 0] = cos.squeeze(1)
        a[:, 0, 1] = -sin.squeeze(1)
        a[:, 1, 0] = sin.squeeze(1)
        a[:, 1, 1] = cos.squeeze(1)

        # нулевой сдвиг: [B, 2, 1]
        zeros = torch.zeros(
            batch_size,
            2,
            1,
            device=tan_theta.device,
            dtype=tan_theta.dtype,
        )

        # theta = [R | 0], shape [B, 2, 3]
        theta = torch.cat([a, zeros], dim=2)

        return theta, a, predicted_angles
