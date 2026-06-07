import torch
import torch.nn as nn
import torch.nn.functional as F


class ROTViTLoss(nn.Module):
    def __init__(
        self,
        w_1=1.0,
        w_2=1.0,
        w_f=0.1,
        w_l=0.1,
        detach_reference=False,
    ):
        super().__init__()

        self.w_1 = w_1
        self.w_2 = w_2
        self.w_f = w_f
        self.w_l = w_l
        self.detach_reference = detach_reference

    def forward(self, logits_per_branch, features_per_branch, targets):
        """
        logits_per_branch:   [2, B, num_classes]
        features_per_branch: [2, B, 197, hidden_size]
        targets:             [B]

        L =
            w_1 * CE(y, l_1)
          + w_2 * CE(y, l_2)
          + w_f * L1(z_1^L, z_2^L)
          + w_l * L1(l_1, l_2)
        """

        logits_1 = logits_per_branch[0]
        logits_2 = logits_per_branch[1]

        features_1 = features_per_branch[0]
        features_2 = features_per_branch[1]

        if self.detach_reference:
            features_1_for_l1 = features_1.detach()
            logits_1_for_l1 = logits_1.detach()
        else:
            features_1_for_l1 = features_1
            logits_1_for_l1 = logits_1

        ce_1 = F.cross_entropy(logits_1, targets)
        ce_2 = F.cross_entropy(logits_2, targets)

        features_l1 = F.l1_loss(features_2, features_1_for_l1)
        logits_l1 = F.l1_loss(logits_2, logits_1_for_l1)

        total_loss = self.w_1 * ce_1 + self.w_2 * ce_2 + self.w_f * features_l1 + self.w_l * logits_l1

        loss_dict = {
            "loss": total_loss.detach(),
            "ce_1": ce_1.detach(),
            "ce_2": ce_2.detach(),
            "features_l1": features_l1.detach(),
            "logits_l1": logits_l1.detach(),
        }

        return total_loss, loss_dict
