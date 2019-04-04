import os.path as osp
import argparse, time, math
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, ChebConv
from ComputeWLvec import computeWL

cache={}
def transform(data):
    if data not in cache:
        #data.z = torch.Tensor(computeWL([data]).todense())
        cache[data]=torch.Tensor(computeWL([data]).todense())
        data.z = cache[data]
    else:
        data.z = cache[data]
    return data

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'NCI1')
dataset = TUDataset(path, name='NCI1', transform=transform).shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]

pretrain_loader = DataLoader(dataset, batch_size=50)

test_loader = DataLoader(test_dataset, batch_size=50)
train_loader = DataLoader(train_dataset, batch_size=50)

class GCNNet(torch.nn.Module):
    def __init__(self, dim=32, gdim=128, pretr_out_dim=100, out_dim=2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, dim, improved=True)
        self.conv2 = GCNConv(dim, dim, improved=True)
        self.conv3 = GCNConv(dim, dim, improved=True)
        #self.conv4 = GCNConv(dim, dim, improved=False)

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.fc1 = Linear(3*dim, gdim)
        self.fc1a = Linear(gdim, gdim)

        self.fc2 = Linear(gdim, pretr_out_dim)
        self.fc3 = Linear(gdim, dataset.num_classes)

    def forward(self, x, edge_index, batch, pretr=False):
        x1 = F.relu(self.conv1(x, edge_index))
        #x1 = F.dropout(x1, training=self.training)
        x2 = self.conv2(x1, edge_index)
        #x2 = F.dropout(x2, training=self.training)
        x3 = self.conv3(x2, edge_index)
        #return F.log_softmax(x, dim=1)
        x= torch.cat([x1,x2,x3], dim=1)
        x = F.relu(self.fc1(x))

        x = global_add_pool(x, batch)
        #print(x.shape)
        #x = F.relu(self.fc1a(x))
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.fc2(x)
        if pretr:
            out1 = self.fc2(x)
        #else:
            #x= self.fc2(x)

        x = self.fc3(x)
        x=F.log_softmax(x, dim=-1)
        if pretr:
            return out1, x #F.log_softmax(x, dim=-1)
        else:
            return x


class GINNet(torch.nn.Module):
    def __init__(self, dim=32, pretr_out_dim=200):
        super(GINNet, self).__init__()

        num_features = dataset.num_features
        #dim = dim

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)
        self.fc3 = Linear(dim, pretr_out_dim)


    def forward(self, x, edge_index, batch, pretr=False):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        #x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        #x = self.fc3(x)

        if pretr:
            x = self.fc3(x)
        else:
            #x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc2(x)
            x=F.log_softmax(x, dim=-1)


        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNNet(dim=32, gdim=128).to(device) #GINNet(dim=32) #
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.005)

def pre_and_train(epoch):
    model.train()

    if epoch % 51 ==0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    loss1_all = 0
    loss2_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output1, output2 = model(data.x, data.edge_index, data.batch, pretr=True) #, data.y
        loss1 = F.l1_loss(output1, data.z)
        loss2 = F.nll_loss(output2, data.y)
        loss=loss2+(0.01*loss1)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss1_all += loss1.item() * data.num_graphs
        loss2_all += loss2.item() * data.num_graphs

        optimizer.step()
    return loss_all / len(train_dataset), loss1_all / len(train_dataset),loss2_all / len(train_dataset)


def pre_train(epoch):
    model.train()

    if epoch % 51 ==0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in pretrain_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, pretr=True) #, data.y
        loss = F.l1_loss(output, data.z)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def train(epoch):
    model.train()

    if epoch % 51 ==0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer2.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer2.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

#pre and train
for epoch in range(1, 51):
    pre_and_train_loss, pre_loss, train_loss = pre_and_train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, All Loss: {:.7f}, Pre Loss: {:.7f}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch,pre_and_train_loss,pre_loss, train_loss,
                                                       train_acc, test_acc))



#pretrain
for epoch in range(1, 31):
    train_loss = pre_train(epoch)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '.format(epoch, train_loss))
torch.save(model.state_dict(), "PretrainedModel")

model.load_state_dict(torch.load("PretrainedModel"))

for epoch in range(1, 51):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, #-2
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200, #200,
            help="number of training epochs")
    parser.add_argument("--k", type=int, default=4, #200,
            help="Conv layer width")
    parser.add_argument("--tied", type=bool, default=False, #48
            help="are weights of CNN tied?")
    parser.add_argument("--n-hidden", type=int, default=16, #48
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0, #1
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, # -4
            help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)