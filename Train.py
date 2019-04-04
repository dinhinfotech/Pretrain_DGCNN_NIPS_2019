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
import numpy as np


def train(epoch, device, model, optimizer, train_loader, alpha):
    model.train()
    #if epoch % 51 ==0 and epoch > 1:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output1 = model(data.x, data.edge_index, data.batch, pretr=False) #, data.y
        #print(data.y)
        loss = F.nll_loss(output1, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader, device, model):
    model.eval()
    loss_all = 0

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset), correct / len(loader.dataset)

def main(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)

    dataset = TUDataset(path, name=args.dataset).shuffle()

    print("Num classes", dataset.num_classes)
    #kf=KFold(n_splits=args.kfsplits, random_state=42, shuffle=True)
    accs=[]
    for i in range(args.kfsplits):
    #for train_index, test_index in kf.split(dataset):
        fold=len(dataset)//args.kfsplits
        if i<(args.kfsplits-1):
            test_fold=i
            val_fold=i+1
        else: #i=9
            test_fold=i
            val_fold=0

        if val_fold >= 2  :
            if val_fold < (args.kfsplits-1):
                train_dataset = dataset[0:test_fold * fold] + dataset[(val_fold + 1) * fold:]
            else:
                train_dataset = dataset[0:test_fold * fold]
        elif val_fold==1:
            train_dataset = dataset[(val_fold + 1) * fold:]
        elif val_fold==0:
            train_dataset = dataset[(val_fold + 1) * fold:(test_fold) * fold]

        val_dataset = dataset[val_fold * fold: (val_fold + 1) * fold]
        test_dataset = dataset[test_fold * fold: (test_fold + 1) * fold]


        #for train_index, test_index in kf.split(dataset):
        #print(_train_dataset)
        #train_dataset = torch.utils.data.Subset(_train_dataset,range(int(0.9*len(_train_dataset))))
        #val_dataset = torch.utils.data.Subset(_train_dataset,range(int(0.9*len(_train_dataset)),len(_train_dataset)))
        #train_dataset, validation_dataset, y_train, y_test = train_test_split(_train_dataset, test_size = 0.1, random_state = 43)

        test_loader = DataLoader(test_dataset, batch_size=50)
        val_loader = DataLoader(val_dataset, batch_size=50)

        train_loader = DataLoader(train_dataset, batch_size=50)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.model=="GCN":
            model = GCNNet(input_dim=dataset.num_features, dim=args.node_hidden_dim, gdim=args.graph_hidden_dim,out_dim=dataset.num_classes).to(device)  # GINNet(dim=32) #
        else:
            model = GINNet(input_dim=dataset.num_features, dim=args.node_hidden_dim).to(device)  # GINNet(dim=32) #

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        best_val_loss=1000000000.0
        best_val_acc=0
        #pre and train
        for epoch in range(0, args.epochs):
            train_loss = train(epoch, device, model, optimizer, train_loader, args.alpha)
            #_, train_acc = test(train_loader, device, model)
            val_loss, val_acc = test(val_loader, device, model)
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch, train_loss,
                                                           val_loss, val_acc))
            #-test
            loss, acc = test(test_loader, device, model)
            accs.append(acc)
            print("Test Accuracy {:.4f}".format(acc))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving model. ValLoss: ", best_val_loss)
                torch.save(model.state_dict(), "MyModel")

        model.load_state_dict(torch.load("MyModel"))
        loss, acc = test(val_loader, device, model)
        print("Best val loss:", loss, "Best Val Acc:", acc)

        print()
        loss, acc = test(test_loader, device, model)
        accs.append(acc)
        print("Test Accuracy {:.4f}".format(acc))
    print("CV results", np.mean(accs), "+-", np.std(accs))


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
    parser.add_argument("--k", type=int, default=4, #200,
            help="Conv layer width")
    parser.add_argument("--node_hidden_dim", type=int, default=32, #48
            help="number of hidden gcn units")
    parser.add_argument("--graph_hidden_dim", type=int, default=128, #48
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0, #1
            help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=0, # 5e-4
            help="Weight for L2 loss")
    parser.add_argument("--kfsplits", type=int, default=10, # -4
            help="Number of splits of kfold")
    parser.add_argument("--alpha", type=float, default=0.01, # -4
            help="Parameter balancing losses")
    parser.add_argument('--pretrain', dest='pretraining', action='store_true', help="Use pretrained model")
    parser.add_argument('--no-pretrain', dest='pretraining', action='store_false', help="Don't use pretrained model (default)")
    parser.set_defaults(pretraining=False)
    args = parser.parse_args()
    print(args)
    main(args)