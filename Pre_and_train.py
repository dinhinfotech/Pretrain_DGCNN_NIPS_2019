import os.path as osp
import argparse, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, ChebConv
from ComputeWLvec import computeWL
from MyNets import GCNNet, GINNet, GCNNetSortPooling
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
cache={}
import shutil

"""Model selection"""


def model_selection(Mtr=None, labels_tr=None, svm_paras=None):
    skf = StratifiedKFold(n_splits=5)
    dict_para_acc = {}
    for svm_idx, svm_para in enumerate(svm_paras):
        dict_para_acc[svm_para] = 0

    for list_tr_idx, list_te_idx in skf.split(np.zeros(len(labels_tr)), labels_tr):
        Mtr_vali = Mtr[np.ix_(list_tr_idx)]
        Mte_vali = Mtr[np.ix_(list_te_idx)]
        labels_tr_vali = [labels_tr[idx] for idx in list_tr_idx]
        labels_te_vali = [labels_tr[idx] for idx in list_te_idx]

        for svm_idx, svm_para in enumerate(svm_paras):
            # clf = svm.SVC(C = svm_para, kernel='rbf')
            clf = LinearSVC(C=svm_para, dual=False, max_iter=50000)
            clf.fit(Mtr_vali, labels_tr_vali)

            y_predict = clf.predict(Mte_vali)
            acc = accuracy_score(labels_te_vali, y_predict)
            dict_para_acc[svm_para] += acc
    # Return optimal C
    return max(dict_para_acc, key=dict_para_acc.get)

def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """

    f = open(file_path, 'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines

from sklearn.preprocessing import normalize
def transform(data):
    if data not in cache:
        #data.z = torch.Tensor(computeWL([data]).todense())
        #print(data)
        if args.knorm:
            cache[data] = torch.Tensor(normalize(computeWL([data], h=args.h, hash=args.hash, separated_iterations=args.separated_iterations)).todense())
        else:
            cache[data]=torch.Tensor(computeWL([data], h=args.h, hash=args.hash, separated_iterations=args.separated_iterations).todense())

        data.z = cache[data]
    else:
        data.z = cache[data]
    return data

def get_hidden(device, model, train_loader):
    hidden_gcn=[]
    hidden_dggcn=[]

    for data in train_loader:
        #print("num graphs", data.num_graphs)
        data = data.to(device)
        hidden_batch_gcn,hidden_batch_dggcn  = model.get_hidden(data.x, data)
        #print(hidden_batch_gcn.shape)
        hidden_gcn.append(hidden_batch_gcn)
        hidden_dggcn.append(hidden_batch_dggcn)

    hmatgcn=torch.cat(hidden_gcn, dim=0).detach().cpu().numpy()
    hmatdggcn=torch.cat(hidden_dggcn, dim=0).detach().cpu().numpy()

    #print(hmatgcn.shape)
    return hmatgcn, hmatdggcn



def pre_and_train(epoch, device, model, optimizer, train_loader, alpha):
    model.train()

    if epoch % 50 ==0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
        alpha=alpha*0.5

    loss_all = 0
    loss1_all = 0
    loss2_all = 0

    for data in train_loader:
        #print("num graphs", data.num_graphs)
        data = data.to(device)
        optimizer.zero_grad()
        output1, output2 = model(data.x, data, pretr=True) #, data.y
        loss1 = F.mse_loss(output1, data.z)
        loss2 = F.nll_loss(output2, data.y)
        loss=(alpha*loss1)+(1.0-alpha)*loss2
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss1_all += loss1.item() * data.num_graphs
        loss2_all += loss2.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset), loss1_all / len(train_loader.dataset),loss2_all / len(train_loader.dataset)


def test(loader, device, model):
    model.eval()
    loss_all = 0

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data)
        loss = F.nll_loss(output, data.y)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_all += loss.item() * data.num_graphs

    return loss_all / len(loader.dataset), correct / len(loader.dataset)

def main(args):
    random.seed(1)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
    print(path)
    shutil.rmtree(path, ignore_errors=True)

    dataset = TUDataset(path, name=args.dataset, pre_transform=transform).shuffle()


    if args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in dataset])
        args.sortpooling_k = num_nodes_list[int(math.ceil(args.sortpooling_k * len(num_nodes_list))) - 1]
        args.sortpooling_k = max(10, args.sortpooling_k)
        print('k used in SortPooling is: ' + str(args.sortpooling_k))

    #kf=KFold(n_splits=args.kfsplits, random_state=42, shuffle=True)
    kfolds=[]
    kfolds_cnn=[]
    kfolds_dgcnn=[]

    from torch_geometric.data import Dataset
    class MyDataset(Dataset):
        def __init__(self,
                     data_list):
            self.data_list = data_list
            super(MyDataset, self).__init__("./_temp")
        def __getitem__(self,idx):
            return self.data_list[idx]
        def _download(self):
            pass
        def _process(self):
            pass
        def __len__(self):
            return len(self.data_list)


    skf = StratifiedKFold(n_splits=10)

    for run in range(1,11):
        accs = []
        accs_cnn = []
        accs_dgcnn = []


        parameters_save = []
        if args.dataset=="PTC_MR":
            dset="PTC"
        else:
            dset=args.dataset
        random_idx = [int(idx) for idx in
                      load_list_from_file("shuffle_idx" + '/' + dset + "_" + str(run))]
        graphs_shuffle = [dataset[idx] for idx in random_idx]
        graphs = graphs_shuffle[:]
        labels = [g.y for g in graphs]
        #print(labels)
        fold=-1
        for list_tr_idx, list_te_idx in skf.split(np.zeros(len(labels)), labels):
            fold+=1
            test_dataset = MyDataset([graphs[idx] for idx in list_te_idx])
            train_dataset = MyDataset([graphs[idx] for idx in list_tr_idx[:-len(list_te_idx)]])

            val_dataset = MyDataset([graphs[idx] for idx in list_tr_idx[-len(list_te_idx):]])

            train_dataset_all = MyDataset([graphs[idx] for idx in list_tr_idx])
            train_dataset_all_labels = [graphs[idx].y for idx in list_tr_idx]
            test_dataset_labels = [graphs[idx].y for idx in list_te_idx]


            test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=50)
            train_all_loader = DataLoader(train_dataset_all, batch_size=50, shuffle=False)

            train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if args.model=="GCN":
                model = GCNNet(input_dim=dataset.num_features, dim=args.node_hidden_dim, gdim=args.graph_hidden_dim, out_dim=dataset.num_classes, pretr_out_dim=args.hash).to(device)  # GINNet(dim=32) #
            elif args.model=="GIN":
                model = GINNet(input_dim=dataset.num_features, dim=args.node_hidden_dim, out_dim=dataset.num_classes, pretr_out_dim=args.hash).to(device)  # GINNet(dim=32) #
            else:
                model = GCNNetSortPooling(input_dim=dataset.num_features, dim=args.node_hidden_dim, gdim=args.graph_hidden_dim,out_dim=dataset.num_classes, pretr_out_dim=args.hash,k=args.sortpooling_k,device=device).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
            best_val_loss=1000000000.0
            best_val_acc=0
            #pre and train
            elapsed_patience=0
            for epoch in range(0, args.epochs):
                if elapsed_patience > args.patience:
                    break
                pre_and_train_loss, pre_loss, train_loss = pre_and_train(epoch, device, model, optimizer, train_loader, args.alpha)
                #_, train_acc = test(train_loader, device, model)
                val_loss, val_acc = test(val_loader, device, model)
                print('Epoch: {:03d}, All Loss: {:.7f}, Pre Loss: {:.7f}, Train Loss: {:.7f}, '
                      'Val Loss: {:.7f}, Val Acc: {:.7f}'.format(epoch,pre_and_train_loss,pre_loss, train_loss,
                                                               val_loss, val_acc))
                #-test
                loss, acc = test(test_loader, device, model)
                #accs.append(acc)
                print("Test Accuracy {:.4f}".format(acc))
                if val_loss < best_val_loss:
                    elapsed_patience=0
                    best_val_loss = val_loss
                    print("Saving model. ValLoss: ", best_val_loss)
                    torch.save(model.state_dict(), args.dataset+"_MyModel")
                else:
                    elapsed_patience+=1

            model.load_state_dict(torch.load(args.dataset+"_MyModel"))
            loss, acc = test(val_loader, device, model)
            print("Best val loss:", loss, "Best Val Acc:", acc)

            print()
            loss, acc = test(test_loader, device, model)
            accs.append(acc)
            print(str(fold)+" fold Test Accuracy {:.4f}".format(acc))
            #compute SVM on hidden representation
            svm_paras = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1e+4]
            Mtr_cnn, Mtr_dgcnn=get_hidden(device, model, train_all_loader)
            Mte_cnn, Mte_dgcnn=get_hidden(device, model, test_loader)

            """Processing with SVM applying at the middle of DGCNN """
            tr_list=[]
            c_opt_cnn = model_selection(Mtr=Mtr_cnn, labels_tr=train_dataset_all_labels, svm_paras=svm_paras)
            # clf = svm.SVC(C = c_opt_dgcnn, kernel='rbf')
            clf = LinearSVC(C=c_opt_cnn, dual=False, max_iter=50000)
            clf.fit(Mtr_cnn, train_dataset_all_labels)

            y_predict_cnn = clf.predict(Mte_cnn)
            acc_cnn = accuracy_score(test_dataset_labels, y_predict_cnn)
            accs_cnn.append(acc_cnn)

            print("ACC SVM CNN",acc_cnn)
            #----------------------------

            """Processing with SVM applying at the end of DGCNN """
            tr_list=[]
            c_opt_dgcnn = model_selection(Mtr=Mtr_dgcnn, labels_tr=train_dataset_all_labels, svm_paras=svm_paras)
            # clf = svm.SVC(C = c_opt_dgcnn, kernel='rbf')
            clf = LinearSVC(C=c_opt_dgcnn, dual=False, max_iter=50000)
            clf.fit(Mtr_dgcnn, train_dataset_all_labels)

            y_predict_dgcnn = clf.predict(Mte_dgcnn)
            acc_dgcnn = accuracy_score(test_dataset_labels, y_predict_dgcnn)
            accs_dgcnn.append(acc_dgcnn)

            print("ACC SVM DGCNN",acc_dgcnn)
            #----------------------------

            #print("Partial CV results", np.mean(accs), "+-", np.std(accs))

        print(str(run)+" CV results", np.mean(accs), "+-", np.std(accs))
        kfolds.append(np.mean(accs))
        kfolds_cnn.append(np.mean(accs_cnn))
        kfolds_dgcnn.append(np.mean(accs_dgcnn))
        print("Temp CV results", np.mean(kfolds), "+-", np.std(kfolds))
        print("Temp CV results SVM GCN", np.mean(kfolds_cnn), "+-", np.std(kfolds_cnn))
        print("Temp CV results SVM DGCNN", np.mean(kfolds_dgcnn), "+-", np.std(kfolds_dgcnn))

    print("CV results", np.mean(kfolds), "+-", np.std(kfolds))
    print("CV results SVM GCN", np.mean(kfolds_cnn), "+-", np.std(kfolds_cnn))
    print("CV results SVM DGCNN", np.mean(kfolds_dgcnn), "+-", np.std(kfolds_dgcnn))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--model", type=str, default="GCNSortPooling", # -4
            help="Graph Conv Model")
    parser.add_argument("--dataset", type=str, default="NCI1", # -4
            help="Name of the datasetto consider")
    parser.add_argument("--lr", type=float, default=1e-2, #-2
            help="learning rate")
    parser.add_argument("--patience", type=int, default=50, #-2
            help="patience")
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
    parser.add_argument("--weight-decay", type=float, default=5e-4, # -4
            help="Weight for L2 loss")
    parser.add_argument("--alpha", type=float, default=0.01, # -4
            help="Parameter balancing losses")
    parser.add_argument("--kfsplits", type=int, default=10, # -4
            help="Number of splits of kfold")
    parser.add_argument("--hash", type=int, default=200, # -4
            help="Hash size")
    parser.add_argument("--h", type=int, default=3,
            help="Kernel h parameter")
    parser.add_argument('--knorm', dest='knorm', action='store_true', help="Use normalized kernel for pretraining")
    parser.add_argument('--no-knorm', dest='knorm', action='store_false', help="Don't normalize kernel for pretraining (default)")
    parser.set_defaults(knorm=False)
    parser.add_argument('--separated-iterations', dest='separated_iterations', action='store_true', help="Separated hashs for WL features (hash should be multiple of graph_size+1)")
    parser.set_defaults(separated_iterations=False)
    parser.add_argument('-sortpooling_k', type=float, default=0.6, help='number of nodes kept after SortPooling')

    args = parser.parse_args()
    print(args)
    main(args)
