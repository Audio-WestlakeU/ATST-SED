import torch.nn as nn
import torch
from .beats.beats_model import BEATs_model
from .RNN import BidirectionalGRU
from .CNN import CNN

class CRNN(nn.Module):
    def __init__(
        self,
        unfreeze_atst_layer=0,
        n_in_channel=1,
        nclass=10,
        activation="glu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        embedding_size=768,
        model_init=None,
        atst_dropout=0.0,
        mode=None,
        **kwargs,
    ):
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.atst_dropout = atst_dropout
        n_in_cnn = n_in_channel
        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
        self.softmax = nn.Softmax(dim=-1)

        self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in)
        
        self.init_beats()
        self.init_model(model_init, mode=mode)
        
        self.unfreeze_atst_layer = unfreeze_atst_layer

    def init_beats(self, path=None):
        self.BEATs_model = BEATs_model()
    
    def init_model(self, path, mode=None):
        if path is None:
            pass
        else:
            if mode == "teacher":
                print("Loading teacher from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
            else:
                print("Loading student from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_student"]
            self.load_state_dict(state_dict, strict=True)
            print("Model loaded")

    def forward(self, x, pretrain_x, pad_mask=None, embeddings=None):
        x = x.transpose(1, 2).unsqueeze(1)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        # rnn features
        embeddings = self.BEATs_model(pretrain_x)
        embeddings = torch.nn.functional.adaptive_avg_pool1d(embeddings.transpose(-1, -2), 156).transpose(-1, -2)
        x = self.cat_tf(torch.cat((x, embeddings), -1))
        
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]

        return strong.transpose(1, 2), weak
