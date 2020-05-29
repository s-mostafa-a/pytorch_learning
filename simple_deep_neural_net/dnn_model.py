from torch import nn
from torch.nn import functional as F


class DNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.output = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        before_activation_func = self.hidden(xb)
        after_activation_func = F.relu(before_activation_func)
        return self.output(after_activation_func)
