import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, LGConv, TAGConv, SGConv,JumpingKnowledge
from graph_global_attention_layer import LowRankAttention, weight_init
# from lg_layer import SAGEConv as SAGE2


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, normalize=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, normalize=True))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, normalize=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class TAGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(TAGC, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TAGConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TAGConv(hidden_channels, hidden_channels))
        self.convs.append(
            TAGConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SGC, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SGConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels))
        self.convs.append(
            SGConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SGCRes(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, K=3, node_pred=False):
        super(SGCRes, self).__init__()

        self.K = K
        # self.node_embeddings = torch.nn.ModuleList()
        # # node embeddings
        # for i in range(num_layers):
        #     first_channels = in_channels if i == 0 else hidden_channels
        #     second_channels = hidden_channels
        #     self.node_embeddings.append(torch.nn.Linear(first_channels, second_channels))
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SGConv(in_channels, out_channels, self.K))
        # for _ in range(num_layers - 2):
        #     self.convs.append(
        #         SGConv(hidden_channels, hidden_channels, self.K))
        # self.convs.append(
        #     SGConv(hidden_channels, out_channels, self.K))
        # self.lins = torch.nn.ModuleList()
        # self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        # for _ in range(num_layers - 2):
        #     self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        # self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.activation = nn.ReLU()

        self.node_pred = node_pred

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for lin in self.lins:
        #     lin.reset_parameters()

    def forward(self, x, adj_t):
        # node embedding
        # for i in range(len(self.node_embeddings)):
        #     x = self.node_embeddings[i](x)
        #     x = self.activation(x)
        # x_res = x
        # for i in range(len(self.convs[:-1])):
        #     x = self.convs[i](x, adj_t)# + self.lins[i](x_res)
        #     x = self.activation(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)# + self.lins[-1](x_res)
        return x.log_softmax(dim=-1) if self.node_pred else torch.sigmoid(x)

class SGCOfficial(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, K=3, node_pred=False):
        super(SGCRes, self).__init__()

        self.K = K
        # self.node_embeddings = torch.nn.ModuleList()
        # # node embeddings
        # for i in range(num_layers):
        #     first_channels = in_channels if i == 0 else hidden_channels
        #     second_channels = hidden_channels
        #     self.node_embeddings.append(torch.nn.Linear(first_channels, second_channels))
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SGConv(in_channels, hidden_channels, self.K))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels, self.K))
        self.convs.append(
            SGConv(hidden_channels, out_channels, self.K))
        # self.lins = torch.nn.ModuleList()
        # self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        # for _ in range(num_layers - 2):
        #     self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        # self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.activation = nn.ReLU()

        self.node_pred = node_pred

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for lin in self.lins:
        #     lin.reset_parameters()

    def forward(self, x, adj_t):
        # node embedding
        # for i in range(len(self.node_embeddings)):
        #     x = self.node_embeddings[i](x)
        #     x = self.activation(x)
        # x_res = x
        for i in range(len(self.convs[:-1])):
            x = self.convs[i](x, adj_t)# + self.lins[i](x_res)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)# + self.lins[-1](x_res)
        return x.log_softmax(dim=-1) if self.node_pred else torch.sigmoid(x)

class SGCwithJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SGCwithJK, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SGConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SGConv(hidden_channels, hidden_channels))
        self.convs.append(
            SGConv(hidden_channels, out_channels))
        
        self.jk = JumpingKnowledge(mode='max', channels=hidden_channels, num_layers=num_layers)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.jk.reset_parameters()


    def forward(self, x, adj_t):
        out_list = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out_list += [x]
        x = self.convs[-1](x, adj_t)
        out_list += [x]
        out = self.jk(out_list)
        return out

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SAGEwithJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGEwithJK, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.jk = JumpingKnowledge(mode='max', channels=hidden_channels, num_layers=num_layers)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.jk.reset_parameters()   

    def forward(self, x, adj_t):
        out_list = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            out_list += [x]
        x = self.convs[-1](x, adj_t)
        out_list += [x]
        out = self.jk(out_list)
        return out

class GCNWithAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCNWithAttention, self).__init__()
        self.k = 100 
        self.hidden = hidden_channels
        self.num_layer = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.attention = torch.nn.ModuleList()
        self.dimension_reduce = torch.nn.ModuleList()
        self.attention.append(LowRankAttention(self.k, in_channels, dropout))
        self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_channels,\
        hidden_channels),nn.ReLU()))
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])      
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.attention.append(LowRankAttention(self.k,hidden_channels, dropout))
            self.dimension_reduce.append(nn.Sequential(nn.Linear(2*self.k + hidden_channels,\
            hidden_channels)))
        self.dimension_reduce[-1] = nn.Sequential(nn.Linear(2*self.k + hidden_channels,\
            out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for glob_attention in self.attention:
            glob_attention.apply(weight_init)
        for dim_reduce in self.dimension_reduce:
            dim_reduce.apply(weight_init)
        for batch_norm in self.bn:
            batch_norm.reset_parameters()

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x_local = F.relu(conv(x, adj))
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            x_global = self.attention[i](x)
            x = self.dimension_reduce[i](torch.cat((x_global, x_local),dim=1))
            x = F.relu(x)
            x = self.bn[i](x)
        x_local = F.relu(self.convs[-1](x, adj))
        x_local = F.dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention[-1](x)
        x = self.dimension_reduce[-1](torch.cat((x_global, x_local),dim=1))
        return x

