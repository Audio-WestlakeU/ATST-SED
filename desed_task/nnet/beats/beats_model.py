import torch
from .BEATs import BEATs, BEATsConfig

class BEATs_model(torch.nn.Module):
    def __init__(self):
        super(BEATs_model, self).__init__()
        checkpoint = torch.load("/home/shaonian/SED/ultra-SED/nnet/ckpts/BEATs_iter3_plus_AS2M.pt", map_location="cpu")
        cfg = BEATsConfig(checkpoint['cfg'])
        self.encoder = BEATs(cfg)
        self.encoder.load_state_dict(checkpoint['model'])
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, beats_feat, other_emb=None):
        return self.encoder.extract_features(beats_feat, None)[0]