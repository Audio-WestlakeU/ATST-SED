import os
import torch
from .audio_transformer import FrameASTModel

class ATST(torch.nn.Module):
    def __init__(self, atst_path, *args, atst_dropout=0.0, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        self.atst = FrameASTModel(atst_dropout=atst_dropout)
        self.load_atst(atst_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, atst_feat, other_emb=None):
        atst_feat = atst_feat.unsqueeze(1)
        atst_x = self.atst.get_intermediate_layers(
            atst_feat,
            self.fake_length.to(atst_feat).repeat(len(atst_feat)),
            1,
            scene=False,
            other_emb=other_emb,
        )
        atst_x = atst_x.transpose(1, 2)
        return atst_x


    def load_atst(self, path=None):
        if path is None:
            pre_path = "../nnet/ckpts/atst_as2M.ckpt" 
            assert os.path.exists(pre_path), "Please make sure you have a default path to load ATST. Please change this path to the atst_as2M.ckpt that you downloaded."
            path = pre_path    # Change path to the atst_as2M.ckpt the downloaded checkpoint from the home page.
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        atst_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                # new_k = k.replace("model.teacher.encoder.", "")
                atst_state_dict[new_k] = v 
            if "encoder.encoder." in k:
                new_k = k.replace("encoder.encoder.", "")
                atst_state_dict[new_k] = v
        self.atst.load_state_dict(atst_state_dict, strict=True)
        for n, param in self.atst.named_parameters():
            param.requires_grad = False
        # state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
        # atst_state_dict = {}
        # for k, v in state_dict.items():
        #     if ("total_ops" in k) or ("total_params" in k):
        #         continue
        #     if "atst_frame.atst." in k:
        #         k = k.replace("atst_frame.atst.", "")
        #         atst_state_dict[k] = v
        # self.atst.load_state_dict(atst_state_dict, strict=True)
        # for n, param in self.atst.named_parameters():
        #     param.requires_grad = False
