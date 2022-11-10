import torch
from torch import nn


# x[t],(h[t-1],c[t-1]) -> h[t],(h[t],c[t])
# x[t]:[batch,x_features] -> h[t]:[batch,hidden_features]
# h[t-1]:[batch,hidden_features] -> h[t]:[batch,hidden_features]
# c[t-1]:[batch,hidden_features] -> c[t]:[batch,hidden_features]
class TextLSTMUnit(nn.Module):
    def __init__(self, x_features, hidden_features):
        super(TextLSTMUnit, self).__init__()

        # i,f,o gate
        self.W_ii = nn.Linear(in_features=x_features, out_features=hidden_features, bias=True)
        self.W_hi = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True)
        self.act_i = nn.Sigmoid()

        self.W_if = nn.Linear(in_features=x_features, out_features=hidden_features, bias=True)
        self.W_hf = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True)
        self.act_f = nn.Sigmoid()

        self.W_io = nn.Linear(in_features=x_features, out_features=hidden_features, bias=True)
        self.W_ho = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True)
        self.act_o = nn.Sigmoid()

        self.W_ig = nn.Linear(in_features=x_features, out_features=hidden_features, bias=True)
        self.W_hg = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True)
        self.act_g = nn.Tanh()

        self.act_c = nn.Tanh()

    def forward(self, x, h, c):
        """
        only one cell
        :param x: [batch,x_features]
        :param h: [batch,hidden_features]
        :param c: [batch,hidden_features]
        :return: h_next,h_next,c_next all is　[batch,hidden_features]
        """
        i = self.act_i(self.W_ii(x) + self.W_hi(h))
        f = self.act_f(self.W_if(x) + self.W_hf(h))
        o = self.act_o(self.W_io(x) + self.W_ho(h))

        g = self.act_g(self.W_ig(x) + self.W_hg(h))
        # i,f,o,g:[batch,hidden_features]

        c_next = f * c + i * g
        h_next = o * self.act_c(c_next)
        # all is [batch,hidden_features]
        return h_next, h_next, c_next


# x[1:n],(h[0],c[0]) -> h[1:n],(h[n],c[n])
# x[1:n]:[batch,seq,x_features] -> h[1:n]:[batch,seq,hidden_features]
# h[0]:[batch,hidden_features] -> h[n]:[batch,hidden_features]
# c[0]:[batch,hidden_features] -> c[n]:[batch,hidden_features]
class TextLSTMLayer(nn.Module):
    def __init__(self, x_features, hidden_features):
        super(TextLSTMLayer, self).__init__()
        self.one_step = TextLSTMUnit(x_features, hidden_features)

    def forward(self, x, h, c):
        seq_len = x.shape[1]
        y = []
        for i in range(seq_len):
            h, h, c = self.one_step(x[:, i, :], h, c)
            y.append(h)

        # y:[tensor(batch,hidden_features)*seq]

        y = torch.cat([i.unsqueeze(0) for i in y], dim=0)

        # y:[seq,batch,hidden_features]
        y = torch.transpose(y, 0, 1)
        # y:[batch,seq,hidden_features]
        return y, h, c


# input,(h_0,c_0) -> output,(h_n,c_n)
# input:[batch,seq,in_features] -> output:[batch,seq,hidden_features]
# h_0:[batch,num_layers,hidden_features](defaults to zeros) -> h_n:[batch,num_layers,hidden_features]
# c_0:[batch,num_layers,hidden_features](defaults to zeros) -> c_n:[batch,num_layers,hidden_features]
class TextLSTM(nn.Module):
    def __init__(self, in_freatures, hidden_features, num_layers=3):
        super(TextLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList([TextLSTMLayer(in_freatures, hidden_features)]
                                    + [TextLSTMLayer(hidden_features, hidden_features) for _ in range(num_layers - 1)])

    def forward(self, inp, h0=None, c0=None):
        """
        :param inp: [batch,seq,in_features]
        :param h0: [batch,num_layers,hidden_features](defaults to zeros)
        :param c0: [batch,num_layers,hidden_features](defaults to zeros)
        :return: output,h_n,c_n
        output:[batch,seq,hidden_features]
        h_n:[batch,num_layers,hidden_features]
        c_n:[batch,num_layers,hidden_features]
        """

        if not h0:
            h0 = torch.zeros(inp.shape[0], self.num_layers, self.hidden_features)

        if not c0:
            c0 = torch.zeros(inp.shape[0], self.num_layers, self.hidden_features)

        hns = []
        cns = []
        for i in range(self.num_layers):
            inp, hn, cn = self.layers[i](inp, h0[:, i, :], c0[:, i, :])
            hns.append(hn)
            cns.append(cn)
        # hns:[num_layers,batch,hidden_features]
        # cns:[num_layers,batch,hidden_features]

        h_n = torch.cat([i.unsqueeze(0) for i in hns], dim=0)
        c_n = torch.cat([i.unsqueeze(0) for i in cns], dim=0)

        return inp, torch.transpose(h_n, 0, 1), torch.transpose(c_n, 0, 1)
        # output:[batch,seq,hidden_features]
        # h_n:[batch,num_layers,hidden_features]
        # c_n:[batch,num_layers,hidden_features]
