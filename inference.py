import yaml
import torch
import torch.nn as nn
from torchaudio.transforms import AmplitudeToDB
from desed_task.nnet.CRNN_e2e import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.dataio.datasets_atst_sed import SEDTransform, ATSTTransform, read_audio
from desed_task.utils.scaler import TorchScaler
from train.local.classes_dict import classes_labels

class ATSTSEDFeatureExtractor(nn.Module):
    def __init__(self, config):
        self.sed_feat_extractor = SEDTransform(**config["feats"])
        self.scaler = TorchScaler(
                "instance",
                config["scaler"]["normtype"],
                config["scaler"]["dims"],
            )
        self.atst_feat_extractor = ATSTTransform()

    
    def take_log(self, mels):
        amp_to_db = AmplitudeToDB(stype="amplitude")
        amp_to_db.amin = 1e-5  # amin= 1e-5 as in librosa
        return amp_to_db(mels).clamp(min=-50, max=80)  # clamp to reproduce old code
    
    def forward(self, mixture):
        mixture = mixture.unsqueeze(0)  # fake batch size
        sed_feats = self.sed_feat_extractor(mixture)
        sed_feats = self.scaler(self.take_log(sed_feats))
        atst_feats = self.atst_feat_extractor(mixture)

        return sed_feats, atst_feats

class ATSTSEDInferencer(nn.Module):
    """Inference module for ATST-SED
    """
    def __init__(self, pretrained_path, model_config_path="./confs/stage2.yaml", audio_dur=10, overlap_dur=3):
        super().__init__()

        # Load model configurations
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize model
        self.model = self.load_from_pretrained(pretrained_path, config)
        # Initialize label encoder
        self.label_encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
        )
        # Initialize feature extractor
        self.feature_extractor = ATSTSEDFeatureExtractor(config)
    
        # Initial parameters
        self.audio_dur = audio_dur
        self.overlap_dur = overlap_dur
        self.fs = config["data"]["fs"]   
    
    def load_from_pretrained(self, pretrained_path: str, config: dict):
        # Initializign model
        model = CRNN(
            unfreeze_atst_layer=config["opt"]["tfm_trainable_layers"], 
            **config["net"], 
            model_init=config["ultra"]["model_init"],
            atst_dropout=config["ultra"]["atst_dropout"],
            atst_init=config["ultra"]["atst_init"],
            mode="teacher")
        # Load pretrained ckpt
        state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
        ### get teacher model
        state_dict = {k: v for k, v in state_dict.items() if "teacher" in k}
        model.load_state_dict(state_dict, strict=True)
        return model
    
    def forward(self, wav_file):
        mixture, onset_s, offset_s, padded_indx = read_audio(
                wav_file, False, False, None
            )
        
        # split wav into chunks with overlap
        if mixture // self.fs < self.audio_dur:
            inference_chunks = [mixture]
        else:      
            inference_chunks = self.chunk_wise
    
if __name__ == "__main__":
    inference_model = ATSTSEDInferencer(
        "/data/home/shaonian/works/ATST-SED/train/exp/stage2/c2f/version_0/epoch=249-step=7249.ckpt",
        "/data/home/shaonian/works/ATST-SED/train/confs/stage2_real.yaml")
    sed_results = inference_model()
    
    