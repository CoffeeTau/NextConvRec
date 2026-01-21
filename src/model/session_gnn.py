import torch
import torch.nn as nn
import torch.nn.functional as F

class SessionGNN(nn.Module):
    def __init__(self, hidden_size):
        super(SessionGNN, self).__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(2 * hidden_size, 3 * hidden_size, bias=True)
        self.w_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

    def forward(self, hidden, A):
        """
        hidden: (B, L, D)
        A: (B, L, L) 邻接矩阵（每个序列一个图）
        """
        input_in = torch.bmm(A, hidden)  # (B, L, D)
        inputs = torch.cat([input_in, hidden], dim=2)  # (B, L, 2D)

        gi = self.w_ih(inputs)
        gh = self.w_hh(hidden)

        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        output = newgate + inputgate * (hidden - newgate)
        return output  # (B, L, D)
