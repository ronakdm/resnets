import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, n_layers):
        super(MLP, self).__init__()
        if n_layers > 0:
            self.proj = nn.Linear(in_features, hidden_size)
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
            self.out = nn.Linear(hidden_size, out_features)
        else:
            self.out = nn.Linear(in_features, out_features)
        self.n_layers = n_layers

    def forward(self, x):
        if self.n_layers > 0:
            x = F.relu(self.proj(x))
            for layer in self.layers:
                x = F.relu(layer(x))
        return self.out(x)
    
class MiniCLIP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, n_layers, architecture="miniclip"):
        if architecture != "miniclip":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model MiniCLIP!"
            )
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features, hidden_size, out_features, n_layers)
        self.text_encoder  = MLP(in_features, hidden_size, out_features, n_layers)
        # self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1)) # learnable parameter
        self.scale = 100.

    def forward(self, x, y, sample_weight=None):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(y)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale

        # symmetric loss function
        labels = torch.arange(len(logits)).to(x.get_device())
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        return loss, logits
    
class JointlyCenteredCLIP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, n_layers, architecture="jointclip"):
        if architecture != "jointclip":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model MiniCLIP!"
            )
        super(JointlyCenteredCLIP, self).__init__()
        self.image_encoder = MLP(in_features, hidden_size, out_features, n_layers)
        self.text_encoder  = MLP(in_features, hidden_size, out_features, n_layers)
        # self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1)) # learnable parameter
        self.scale = 100.

    def forward(self, x, y, sample_weight=None):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(y)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale
        norm_factor = torch.logsumexp(torch.flatten(logits), dim=0)
        loss = -torch.mean(torch.diagonal(logits) - norm_factor)

        return loss, logits
    
class DoublyCenteredCLIP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, n_layers, architecture="doubleclip"):
        if architecture != "doubleclip":
            raise ValueError(
                f"Incorrect architecture specification '{architecture}' for model MiniCLIP!"
            )
        super(DoublyCenteredCLIP, self).__init__()
        self.image_encoder = MLP(in_features, hidden_size, out_features, n_layers)
        self.text_encoder  = MLP(in_features, hidden_size, out_features, n_layers)
        # self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1)) # learnable parameter
        self.scale = 100.

    def forward(self, x, y, sample_weight=None):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(y)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale

        cx   = F.log_softmax(logits, dim=1)
        cy   = F.log_softmax(logits, dim=0)
        cycx = F.log_softmax(cx, dim=0)
        cxcy = F.log_softmax(cy, dim=1)
        loss = -torch.mean(0.5 * torch.diagonal(cycx) + 0.5 * torch.diagonal(cxcy))

        return loss, logits