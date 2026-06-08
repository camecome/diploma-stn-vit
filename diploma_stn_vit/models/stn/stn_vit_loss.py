import torch
import torch.nn as nn
import torch.nn.functional as F


class STNViTLoss(nn.Module):
    def __init__(
        self,
        w_1=1.0,
        w_2=1.0,
        w_f=1.0,
        w_l=1.0,
        w_affine=1.0,
        detach_reference=False,
        affine_reg_type="orthogonal",
    ):
        super().__init__()

        self.w_1 = w_1
        self.w_2 = w_2
        self.w_f = w_f
        self.w_l = w_l
        self.w_affine = w_affine

        self.detach_reference = detach_reference
        self.affine_reg_type = affine_reg_type

    def forward(self, logits_per_branch, features_per_branch, theta_per_branch, targets):
        """
        logits_per_branch:   [num_branches, batch_size, 1000]
        features_per_branch: [num_branches, batch_size, 197, 768]
        theta_per_branch:    [num_rotations, batch_size, 6 или 4]
        targets:             [batch_size]
        """

        logits_1 = logits_per_branch[0]  # l_1
        logits_2 = logits_per_branch[1]  # l_2

        features_1 = features_per_branch[0]  # f_1
        features_2 = features_per_branch[1]  # f_2

        # вдруг мы сначала не хотим обучать опорную ветку
        if self.detach_reference:
            features_1_for_l1 = features_1.detach()
            logits_1_for_l1 = logits_1.detach()
            raise NotImplementedError(":(")
        else:
            features_1_for_l1 = features_1
            logits_1_for_l1 = logits_1

        ce_1 = F.cross_entropy(logits_1, targets)
        ce_2 = F.cross_entropy(logits_2, targets)

        features_l1 = F.l1_loss(features_2, features_1_for_l1)
        logits_l1 = F.l1_loss(logits_2, logits_1_for_l1)

        affine_l2 = self.affine_regularization(theta_per_branch)

        total_loss = (
            self.w_1 * ce_1
            + self.w_2 * ce_2
            + self.w_f * features_l1
            + self.w_l * logits_l1
            + self.w_affine * affine_l2
        )

        loss_dict = {
            "loss": total_loss,
            "ce_1": ce_1.detach(),
            "ce_2": ce_2.detach(),
            "features_l1": features_l1.detach(),
            "logits_l1": logits_l1.detach(),
            "affine_l2": affine_l2.detach(),
        }

        return total_loss, loss_dict

    def affine_regularization(self, theta_per_branch):
        """
        Регуляризация только левой квадратной подматрицы A.

        theta_per_branch:
            [num_rotations, B, 2, 2]
        """
        if theta_per_branch.ndim != 4 or theta_per_branch.shape[-2:] != (2, 2):
            raise ValueError(f"Expected theta_per_branch with shape [num_rotations, B, 2, 2], got {theta_per_branch.shape}")

        # a [num_rotations * B, 2, 2]
        a = theta_per_branch.reshape(-1, 2, 2)

        batch_size = a.shape[0]

        identity = torch.eye(
            2,
            device=a.device,
            dtype=a.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        if self.affine_reg_type == "orthogonal":
            # A A^T должно быть близко к I
            aat = torch.bmm(a, a.transpose(1, 2))
            return F.mse_loss(aat, identity)

        raise ValueError(f"Unknown affine_reg_type={self.affine_reg_type}")
