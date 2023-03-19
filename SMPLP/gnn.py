import argparse
import string
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
import torch_geometric.transforms as T

# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from encoder import GCN, TAGC, SGC, GCNWithAttention, SGCwithJK, SAGEwithJK,  GCN, SGCRes
from decoder import LinkPredictor, DotPredictor, MLPCatPredictor, MLPDotPredictor, MLPBilPredictor, BilinearPredictor, InteractionNetPredictor
from logger import Logger

import wandb


def train_edge(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # random element of previously sampled negative edges
        # negative samples are obtained by using spatial sampling criteria

        edge = neg_train_edge[perm].t()
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def train_node(model, predictor, data, split_node, optimizer, batch_size):
    model.train()
    predictor.train()

    train_idx = split_node['train'].to(data.x.device)

    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = total_examples = 0
    for perm in DataLoader(range(train_idx.size(0)), batch_size,
                           shuffle=True):

        optimizer.zero_grad()

        out = model(data.x, data.adj_t)[train_idx[perm]]

        # out = predictor(h)

        # loss = F.nll_loss(out.log_softmax(dim=-1), data.y.squeeze(1)[train_idx])
        loss = criterion(out, data.y[train_idx[perm]].to(torch.float))

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()
        total_examples += 1

    return total_loss / total_examples
@torch.no_grad()
def test_edge(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_rocauc = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_train_pred,
        })[f'rocauc']

    valid_rocauc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
        })[f'rocauc']

    test_rocauc = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

@torch.no_grad()
def test_node(model, predictor, data, split_node, evaluator, batch_size):
    model.eval()
    predictor.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)


    train_acc = evaluator.eval({
            'y_true': data.y[split_node['train']],
            'y_pred': y_pred[split_node['train']]
        })[f'acc']

    valid_acc = evaluator.eval({
        'y_true': data.y[split_node['valid']],
        'y_pred': y_pred[split_node['valid']]
        })[f'acc']

    test_acc = evaluator.eval({
            'y_true': data.y[split_node['test']],
            'y_pred': y_pred[split_node['test']]
        })[f'acc']

    return train_acc, valid_acc, test_acc

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)

def main():
    parser = argparse.ArgumentParser(description='OGBL-VESSEL (GNN) Algorithm.')
    
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--encoder_name', type=str, default='gcn', help='sage, lrga, gcn, sgc, sgcwjk, sagewjk, tagc')
    parser.add_argument('--decoder_name', type=str, default='mlp', help='mlp, dot, mlpcat, mlpdot, mlpbil, bil')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--mlp_dropout', type=float, default=0.4)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--data_name', type=str, default='ogbl-vessel')
    parser.add_argument('--res_dir', type=str, default='log/')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=1e-6) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--experiment_name', type=str, default='exp_0')



    args = parser.parse_args()
    print(args)

    wandb.init(project="sgc",
               entity="cs224wfinal",
               name=args.experiment_name)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if "ogbn-arxiv" in args.data_name:
        dataset = PygNodePropPredDataset(name=args.data_name,
                                         transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
    elif "ogbn-proteins" in args.data_name:
        dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
        data = dataset[0]
        # Move edge features to node features.
        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)
        data.adj_t = data.adj_t.to_symmetric()
    elif "ogbl" in args.data_name:
        dataset = PygLinkPropPredDataset(args.data_name,
                                        transform=T.ToSparseTensor())
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()

    # # normalize x,y,z coordinates  
    # data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
    # data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
    # data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)

    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
    data = data.to(device)

    if "ogbn" in args.data_name:
        split = dataset.get_idx_split()
    elif "ogbl" in args.data_name:
        split = dataset.get_edge_split()

    # create log file and save args
    log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    log_file = os.path.join(args.res_dir, log_file_name)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    out_channel = args.hidden_channels if "ogbl" in args.data_name else args.hidden_channels # dataset.num_classes
    if args.encoder_name.lower() == 'sage':
        model = SAGE(data.num_features, args.hidden_channels,
                     out_channel, args.num_layers,
                     args.dropout).to(device)
    elif args.encoder_name.lower() == 'lrga':
        model = GCNWithAttention(data.num_features, args.hidden_channels,
                     out_channel, args.num_layers,
                     args.dropout).to(device)
    elif args.encoder_name.lower() == 'gcn':
        model = GCN(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sgc':
        model = SGC(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sgcwjk':
        model = SGCwithJK(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sagewjk':
        model = SAGEwithJK(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'tagc':
        model = TAGC(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder_name.lower() == 'sgcres':
        model = SGCRes(data.num_features, args.hidden_channels,
                    out_channel, args.num_layers,
                    args.dropout, node_pred=("ogbn" in args.data_name)).to(device)
    else:
        print('Wrong model name!')

        # # Pre-compute GCN normalization.
        # adj_t = data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # data.adj_t = adj_t
    out_channel = 1 if "ogbl" in args.data_name else dataset.num_classes
    if args.decoder_name.lower() == 'mlp':
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif args.decoder_name.lower() == 'dot':
        predictor = DotPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif  args.decoder_name.lower() == 'mlpcat':
        predictor = MLPCatPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif  args.decoder_name.lower() == 'mlpdot':
        predictor = MLPDotPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif  args.decoder_name.lower() == 'mlpbil':
        predictor = MLPBilPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif  args.decoder_name.lower() == 'bil':
        predictor = BilinearPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    elif args.decoder_name.lower() == 'inet':
        predictor = InteractionNetPredictor(args.hidden_channels, args.hidden_channels, out_channel,
                                args.mlp_layers, args.mlp_dropout).to(device)
    else:
        print('Wrong predictor name!')

    evaluator = Evaluator(name=args.data_name)
    logger = Logger(args.runs, args)   

    # 
    sum_params = 0.
    for p in model.parameters():
        sum_params += p.numel()
    if "ogbl" in args.data_name:
        for p in predictor.parameters():
            sum_params += p.numel()
    print(f'Params: {sum_params}')

    train_roc_auc_list=[]
    valid_roc_auc_list=[]
    for run in range(args.runs):
        set_seed(run)
        
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.AdamW( # AdamW = Adam > SGD
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            if "ogbn" in args.data_name:
                loss = train_node(model, predictor, data, split, optimizer,
                             args.batch_size)
            elif "ogbl" in args.data_name:
                loss = train_edge(model, predictor, data, split, optimizer,
                             args.batch_size)
            if epoch % args.eval_steps == 0:
                if "ogbn" in args.data_name:
                    result = test_node(model, predictor, data, split, evaluator,
                                   args.batch_size)
                elif "ogbl" in args.data_name:
                    result = test_edge(model, predictor, data, split, evaluator,
                                   args.batch_size)
                logger.add_result(run, result)

                train_roc_auc, valid_roc_auc, test_roc_auc = result
                print(f'Run: {run + 1:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.7f}, '
                    f'Train: {train_roc_auc:.7f}, '
                    f'Valid: {valid_roc_auc:.7f}, '
                    f'Test: {test_roc_auc:.7f}')
                train_roc_auc_list.append(train_roc_auc)
                valid_roc_auc_list.append(valid_roc_auc)
                wandb.log({
                    "Loss": loss,
                    "Train": train_roc_auc,
                    "Valid": valid_roc_auc,
                    "Test": test_roc_auc,
                }, step=epoch)
            # if epoch>3 and (valid_roc_auc_list[epoch-1]- valid_roc_auc_list[epoch-2])<1e-6 and valid_roc_auc_list[epoch-1]- valid_roc_auc_list[epoch-3]<1e-6:
            #     print('reset parameters')
            #     model.reset_parameters()
            #     predictor.reset_parameters()
        print('GNN')
        logger.print_statistics(run)

    print('GNN')
    logger.print_statistics()

if __name__ == "__main__":
    main()
