import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_sort_pool, ChebConv
from torch.nn import Sequential, Linear, ReLU

class GCNNetSortPooling(torch.nn.Module):
    def __init__(self, input_dim=0, dim=32, gdim=128, pretr_out_dim=100, out_dim=2, k=100, device="cuda"):
        super(GCNNetSortPooling, self).__init__()
        self.device=device

        self.k=k
        print("Sortpooling k",k)
        self.conv1 = GCNConv(input_dim, dim, improved=False)
        self.conv2 = GCNConv(dim, dim, improved=False)
        self.conv3 = GCNConv(dim, dim, improved=False)
        self.conv4 = GCNConv(dim, 1, improved=False)


        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.sortpooling=global_sort_pool

        #self.fc1 = Linear(self.dense_dim , gdim)
        self.total_conv=3*dim+1

        self.conv1d_params1 = torch.nn.Conv1d(1, 128, self.total_conv, self.total_conv)
        self.maxpool1d = torch.nn.MaxPool1d(2, 2)
        self.conv1d_params2 = torch.nn.Conv1d(128,64, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - 5 + 1) * 64
        #self.fc2 = Linear(self.total_conv, pretr_out_dim)
        self.fc2 = Linear(self.dense_dim, pretr_out_dim)

        self.fc1 = Linear(self.dense_dim , gdim)

        self.fc3 = Linear(gdim , out_dim)
        #self.maxpool1d = nn.MaxPool1d(2, 2)
        #self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

    def get_hidden(self, x, data):
        edge_index=data.edge_index
        batch=data.batch
        x1 = F.tanh(self.conv1(x, edge_index))

        x2 = F.tanh(self.conv2(x1, edge_index))

        x3 = F.tanh(self.conv3(x2, edge_index))
        x4 = F.tanh(self.conv4(x2, edge_index))

        x= torch.cat([x1,x2,x3,x4], dim=1)
        out_temp = global_add_pool(x, batch)

        x=self.sortpooling(x, batch, self.k).to(self.device) # k
        ''' traditional 1d convlution and dense layers '''
        to_conv1d = x.view((-1, 1, self.k * self.total_conv))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        x = conv1d_res.view(data.num_graphs, -1)
        x = F.relu(self.fc1(x))

        return out_temp, x

    def forward(self, x, data, pretr=False):
        edge_index=data.edge_index
        batch=data.batch
        x1 = F.tanh(self.conv1(x, edge_index))

        x2 = F.tanh(self.conv2(x1, edge_index))

        x3 = F.tanh(self.conv3(x2, edge_index))
        x4 = F.tanh(self.conv4(x2, edge_index))

        x= torch.cat([x1,x2,x3,x4], dim=1)
        #out_temp = global_add_pool(x, batch)
        #out1 = (self.fc2(out_temp))

        x=self.sortpooling(x, batch, self.k).to(self.device) # k
         #torch.sigmoid
        #out1 = (self.fc2(x))
        ''' traditional 1d convlution and dense layers '''
        to_conv1d = x.view((-1, 1, self.k * self.total_conv))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)
        x = conv1d_res.view(data.num_graphs, -1)
        #x=F.dropout(x, training=self.training, p=0.5)
        #print(x.shape)

        out1 = (self.fc2(x))
        #conv1d_res = self.conv1d_params1(to_conv1d)
        #conv1d_res = F.relu(conv1d_res)
        #conv1d_res = self.maxpool1d(conv1d_res)
        #conv1d_res = self.conv1d_params2(conv1d_res)
        #conv1d_res = F.relu(conv1d_res)
         #torch.sigmoid

        x = F.relu(self.fc1(x))
        x=F.dropout(x, training=self.training)
        #x = F.relu(self.fc2(x))

        #x = global_add_pool(x, batch)
        #x=F.dropout(x, training=self.training)
        #out1 = (self.fc2(x))


        x = (self.fc3(x))
        #x=F.dropout(x, training=self.training)

        x=F.log_softmax(x, dim=-1)
        if pretr:
            return out1, x #F.log_softmax(x, dim=-1)
        else:
            return x



class GCNNet(torch.nn.Module):
    def __init__(self, input_dim=0, dim=32, gdim=128, pretr_out_dim=100, out_dim=2):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(input_dim, dim, improved=False)
        self.conv2 = GCNConv(dim, dim, improved=False)
        self.conv3 = GCNConv(dim, dim, improved=False)
        #self.conv4 = GCNConv(dim, dim, improved=False)

        #self.conv4 = GCNConv(dim, dim, improved=False)

        #self.conv4 = GCNConv(dim, dim, improved=False)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        #self.bn4 = torch.nn.BatchNorm1d(dim)


        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.fc1 = Linear(3*dim, gdim//2)
        self.fc2a = Linear(gdim//2, gdim)

        self.fc1a = Linear(gdim, gdim)
        self.fc2b = Linear(gdim, gdim//2)

        self.fc2 = Linear(gdim, pretr_out_dim)
        self.fc3 = Linear(gdim//2, out_dim)

    def forward(self, x, edge_index, batch, pretr=False):
        x1 = F.relu(self.conv1(x, edge_index))
        #x1 = self.bn1(x1)

        #x1 = F.dropout(x1, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index))
        #x2 = self.bn2(x2)

        #x2 = F.dropout(x2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index))
        #x3 = self.bn3(x3)

        #x4 = self.conv4(x3, edge_index)
        #x4 = self.bn4(x4)

        #return F.log_softmax(x, dim=1)
        x= torch.cat([x1,x2,x3], dim=1)
        #x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #x=F.dropout(x, training=self.training)
        x = F.relu(self.fc2a(x))

        x = global_add_pool(x, batch)
        #x=F.dropout(x, training=self.training)
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.fc2(x)
        #if pretr:
        #else:
            #x= self.fc2(x)
        #xtemp = F.relu(x)
        out1 = (self.fc2(x)) #torch.sigmoid

        x = F.relu(self.fc2b(x))
        #x=F.dropout(x, training=self.training)

        x = self.fc3(x)
        x=F.log_softmax(x, dim=-1)
        if pretr:
            return out1, x #F.log_softmax(x, dim=-1)
        else:
            return x


class GINNet(torch.nn.Module):
    def __init__(self, input_dim=0, dim=32, pretr_out_dim=100, out_dim=2):
        super(GINNet, self).__init__()

        num_features = input_dim
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
        self.fc2 = Linear(dim, out_dim)
        self.fc3 = Linear(dim, pretr_out_dim)


    def forward(self, x, data, pretr=False):
        edge_index=data.edge_index
        batch= data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        #x = F.relu(self.conv4(x, edge_index))
        #x = self.bn4(x)
        #x = F.relu(self.conv5(x, edge_index))
        #x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        #x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        #x = self.fc3(x)

        if pretr:
            out1 = self.fc3(x)
        else:
            #x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc2(x)
            x=F.log_softmax(x, dim=-1)


        if pretr:
            return out1, x #F.log_softmax(x, dim=-1)
        else:
            return x


