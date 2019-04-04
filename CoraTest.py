from torch_geometric.datasets import Planetoid
import os.path as osp
import torch_geometric.transforms as T

dataset = 'Citeseer'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())

print(len(dataset))
print(dataset.num_classes)
print(dataset.num_features)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from MonomialGAT import GATConv, MonomialGATConv

class MonomialGATNet(torch.nn.Module):
    def __init__(self):
        super(MonomialGATNet, self).__init__()
        self.att1 = MonomialGATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        self.att2 = MonomialGATConv(8*8 , dataset.num_classes,heads=1, dropout=0.6) #* 8

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.att1(x, data.edge_index)) #F.elu
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.att2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.att1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        self.att2 = GATConv(8 * 8, dataset.num_classes, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.att1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.att2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, K=2):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 16, K)
        self.conv2 = ChebConv(16, dataset.num_classes, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MonomialGATNet()#GCNNet() #ChebNet(K=2)# GATNet()#ChebNet(K=2)#.to(device)
data = dataset[0]#.to(device)
print(data.val_mask)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
best_val_acc=0.0
model.train()
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    #validation set classification
    model.eval()
    _,pred = model(data).max(dim=1)
    correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
    acc = correct / data.val_mask.sum().item()

    print("Epoch {:05d} | Loss {:.4f}  | ValAccuracy {:.4f}".format(epoch, loss.item(),
                                        acc))
    if acc > best_val_acc:
            best_val_acc=acc
            print("Saving model. ValAcc: ", best_val_acc)
            torch.save(model.state_dict(), "MyModel")

model.load_state_dict(torch.load("MyModel"))

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))