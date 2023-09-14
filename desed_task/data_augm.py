import numpy as np
import torch
import random

def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data

def mixup_w_pretrained(feat1, feat2, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    with torch.no_grad():
        batch_size = feat1.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        if mixup_label_type == "soft":
            mixed_feat1 = c * feat1 + (1 - c) * feat1[perm, :]
            mixed_feat2 = c * feat2 + (1 - c) * feat2[perm, :]
            mixed_target = torch.clamp(
                c * target + (1 - c) * target[perm, :], min=0, max=1
            )
        elif mixup_label_type == "hard":
            mixed_feat1 =  (feat1 + feat1[perm, :]) / 2
            mixed_feat2 =  (feat2 + feat2[perm, :]) / 2
            mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
        else:
            raise NotImplementedError(
                f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                f"{'soft', 'hard'}"
            )

        return mixed_feat1, mixed_feat2, mixed_target, perm, c
