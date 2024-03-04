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
    def __init__(self, in_features, hidden_size, out_features, n_layers):
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features, hidden_size, out_features, n_layers)
        self.text_encoder  = MLP(in_features, hidden_size, out_features, n_layers)
        self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1))

    def forward(self, x, y, sample_weight=None):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(y)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.logit_scale.exp()

        # symmetric loss function
        labels = torch.arange(len(logits))
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        return loss, logits