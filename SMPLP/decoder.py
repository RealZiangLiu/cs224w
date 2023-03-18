import torch
import torch.nn as nn
import torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class DotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x

class MLPCatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPCatPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        in_channels = 2 * in_channels
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x1 = torch.cat([x_i, x_j], dim=-1)
        x2 = torch.cat([x_j, x_i], dim=-1)
        for lin in self.lins[:-1]:
            x1, x2 = lin(x1), lin(x2)
            x1, x2 = F.relu(x1), F.relu(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.lins[-1](x1)
        x2 = self.lins[-1](x2)
        x = (x1 + x2)/2
        return x

class InteractionNetPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(InteractionNetPredictor, self).__init__()
        self.node_embeddings = torch.nn.ModuleList()
        self.edge_embeddings = torch.nn.ModuleList()
        self.relation_embeddings = torch.nn.ModuleList()
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        # node embeddings
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = hidden_channels
            self.node_embeddings.append(torch.nn.Linear(first_channels, second_channels))
        # edge embeddings
        for i in range(num_layers):
            first_channels = 2 * hidden_channels if i == 0 else hidden_channels
            second_channels = hidden_channels
            self.edge_embeddings.append(torch.nn.Linear(first_channels, second_channels))
        # relation embeddings
        for i in range(num_layers):
            first_channels = 3 * hidden_channels if i == 0 else hidden_channels
            second_channels = hidden_channels if i < num_layers-1 else out_channels
            self.relation_embeddings.append(torch.nn.Linear(first_channels, second_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.node_embeddings:
            lin.reset_parameters()
        for lin in self.edge_embeddings:
            lin.reset_parameters()
        for lin in self.relation_embeddings:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # node embeddings
        x1 = x_i
        x2 = x_j
        for layer in self.node_embeddings[:-1]:
            x1, x2 = layer(x_i), layer(x_j)
            x1, x2 = self.activation(x1), self.activation(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.node_embeddings[-1](x1)
        x2 = self.node_embeddings[-1](x2)
        # edge embeddings
        x = torch.cat([x1, x2], dim=-1)
        for layer in self.edge_embeddings[:-1]:
            x = layer(x)
            x1 = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.edge_embeddings[-1](x)
        # relation embeddings
        r = torch.cat([x1, x2, x], dim=-1)
        for layer in self.relation_embeddings[:-1]:
            r = layer(r)
            r = F.relu(r)
            r = F.dropout(r, p=self.dropout, training=self.training)
        r = self.relation_embeddings[-1](r)
        return torch.sigmoid(r)


class MLPDotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPDotPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(x_i * x_j, dim=-1)
        return x

class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x

class BilinearPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(BilinearPredictor, self).__init__()
        self.bilin = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x
