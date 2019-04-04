import os.path as osp
import argparse, time, math

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, ChebConv
from ComputeWLvec import computeWL
from MyNets import GCNNet, GINNet
from sklearn.preprocessing import normalize

def transform(data):
    if data not in cache:
        #data.z = torch.Tensor(computeWL([data]).todense())
        #print(data)
        if args.knorm:
            cache[data] = torch.Tensor(normalize(computeWL([data], h=args.h, hash=args.hash)).todense())
        else:
            cache[data]=torch.Tensor(computeWL([data], h=args.h, hash=args.hash).todense())

        data.z = cache[data]
    else:
        data.z = cache[data]
    return data


def pre_train(epoch, device, model, optimizer, train_loader, alpha):
    model.train()

    if epoch % 50 ==0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data.x, data.edge_index, data.batch, pretr=True) #, data.y
        loss = F.mse_loss(output, data.z)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)




def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
    print(path)
    dataset = TUDataset(path, name=args.dataset, pre_transform=transform).shuffle()

    pretrain_loader = DataLoader(dataset, batch_size=50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model=="GCN":
        model = GCNNet(input_dim=dataset.num_features, dim=args.node_hidden_dim, gdim=args.graph_hidden_dim).to(device)  # GINNet(dim=32) #
    else:
        model = GINNet(input_dim=dataset.num_features, dim=args.node_hidden_dim).to(device)  # GINNet(dim=32) #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #pretrain
    for epoch in range(0, args.epochs):
        train_loss = pre_train(epoch, device, model, optimizer, pretrain_loader, args.alpha)
        print('Epoch: {:03d}, Train Loss: {:.7f}, '.format(epoch, train_loss))
    torch.save(model.state_dict(), args.dataset+"-PretrainedModel")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--model", type=str, default="GCN", # -4
            help="Graph Conv Model")
    parser.add_argument("--dataset", type=str, default="NCI1", # -4
            help="Name of the datasetto consider")
    parser.add_argument("--lr", type=float, default=1e-2, #-2
            help="learning rate")
    parser.add_argument("--epochs", type=int, default=201, #200,
            help="number of training epochs")
    parser.add_argument("--k", type=int, default=3, #200,
            help="Conv layer width")
    parser.add_argument("--node_hidden_dim", type=int, default=32, #48
            help="number of hidden gcn units")
    parser.add_argument("--graph_hidden_dim", type=int, default=128, #48
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0, #1
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, # -4
            help="Weight for L2 loss")
    parser.add_argument("--alpha", type=float, default=0.01, # -4
            help="Parameter balancing losses")
    parser.add_argument("--hash", type=int, default=200, # -4
            help="Hash size")
    parser.add_argument("--h", type=int, default=3,
            help="Kernel h parameter")
    args = parser.parse_args()
    print(args)
    main(args)