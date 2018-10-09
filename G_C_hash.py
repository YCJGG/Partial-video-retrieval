import torch
from BatchReader import DatasetProcessing
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import argparse
import networks
from networks import GANLoss, Generator, Discriminator
import torch.optim as optim
import CalcHammingRanking as CalcHR
from torch.optim import lr_scheduler
from sklearn import neighbors
from sklearn import svm



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default= 256, help = "batch size")
parser.add_argument('--g_input_size', type = int, default= 4096, help = "input size of generator")
parser.add_argument('--g_hidden_size', type = int, default= 4096, help = "hidden size of generator")
parser.add_argument('--g_output_size', type = int, default= 4096, help = "output size of generator")
parser.add_argument('--d_input_size', type = int, default= 4096, help = "input size of discriminator")
parser.add_argument('--d_hidden_size', type = int, default= 1024, help = "hidden size of discriminator")
parser.add_argument('--d_output_size', type = int, default= 64 , help = "output size of discriminator")
parser.add_argument('--h_input_size', type = int, default= 4096, help = "input size of Hashnet")
parser.add_argument('--h_hidden_size', type = int, default= 4096, help = "hidden size of Hashnet")
parser.add_argument('--bit', type = int, default= 64 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 3e-5, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 1e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 5e-4, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 150, help = "training epochs")
parser.add_argument('--lamb', type = float, default = 10, help = "lambada")
opt = parser.parse_args()


# load labels
def LoadLabel(filename):
    fp = open(filename,'r')
    labels = [x.strip().split()[1] for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int,labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt
def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S



#dataloader
TRAIN_DIR = 'TrainSplit1.txt'
TEST_DIR = 'TestSplit1.txt'
nclasses = 51


train_data = DatasetProcessing(TRAIN_DIR)
test_data = DatasetProcessing(TEST_DIR)

num_train, num_test = len(train_data) , len(test_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data,batch_size = 1, shuffle = False, num_workers = 1)


train_labels = LoadLabel(TRAIN_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
Y = train_labels_onehot

Sim = CalcSim(train_labels_onehot, train_labels_onehot)


G = networks.Generator(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)
# print(G)
# print(D)
# print(H)

G_dict = G.state_dict()
pretrained_dict = torch.load('./G2_models.pt')
pretrained_dict = pretrained_dict.state_dict()
pretrained_dict = {k : v for k, v in pretrained_dict.items() if  k in G_dict}
G_dict.update(pretrained_dict)
G.load_state_dict(G_dict)



G.cuda()
H.cuda()


criterion = nn.CrossEntropyLoss().cuda()

H_optimizer = optim.SGD(
        H.parameters(),
        lr=opt.lrH,
        weight_decay=1e-5)


#H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

print("###training start~~~~")

# initialize the B and H
FB = torch.zeros(num_train, opt.bit)
FU = torch.zeros(num_train, opt.bit)
PB = torch.zeros(num_train, opt.bit)
PU = torch.zeros(num_train, opt.bit)
max_map = 0
itr = 0
#scheduler = lr_scheduler.StepLR(H_optimizer, step_size=15, gamma=0.1)
for epoch in range(opt.train_epoch):
    epoch_loss = 0.0
    G.eval()
    H.train()
    itt = 0
    for iteration, batch in enumerate(train_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]

        ll = label

        ff, pf = Variable(ff.cuda()), Variable(pf.cuda())

        #label = EncodingOnehot(label, nclasses)
        
        label = Variable(label.cuda())

        ##train_label_onehot = label
        #S = CalcSim(train_label_onehot, train_label_onehot)
        #label = torch.unsqueeze(label,1)
        # generate partial feature to full feature
       
        #pf_cat = torch.cat((pf,label),2)

        uf = torch.randn(pf.size())
        uf = uf.cuda()

        H_optimizer.zero_grad()
    
        #cat = torch.zeros([pf.size()[0],1,nclasses])
        #cat = Variable(cat.cuda())
        #pf_cat_ = torch.cat((pf,cat),2)
        fakef = G(pf, uf)
        #H_fake = H(fakef)
        H_fake, pf_logits = H(fakef)
        H_real, ff_logits = H(ff)
        for i, ind in enumerate(batch_ind):

            FU[batch_ind, :] = H_real.data[i].cpu()
            FB[batch_ind, :] = torch.sign(H_real.data[i].cpu())
            PU[batch_ind, :] = H_fake.data[i].cpu()
            PB[batch_ind, :] = torch.sign(H_fake.data[i].cpu())

        FBbatch = torch.sign(H_real)
        PBbatch = torch.sign(H_fake)

        label = label.view(-1)
        pf_logits = pf_logits.squeeze()
        ff_logits = ff_logits.squeeze()

        clsloss1 = criterion(pf_logits, label)
        #clsloss2 =  criterion(ff_logits, label)

        #regterm1 = (FBbatch-H_real).pow(2).sum() / (num_train * len(label))
        regterm2 = (PBbatch-H_fake).pow(2).sum() /(num_train )
        #regterm3 =  (H_real-H_fake).pow(2).sum() /(num_train * len(label))

        loss =  clsloss1 + 10 *(   regterm2 )
        #loss =  - logloss + 10 * regterm2 
        loss.backward()
        H_optimizer.step()
        epoch_loss += loss.data[0]

    print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch+1, opt.train_epoch, epoch_loss / len(train_loader)))

        # l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
        # print('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l1, l2, t1))




    G.eval()
    H.eval()
    T = np.zeros([num_test,opt.bit],dtype = np.float32)
    FF = np.zeros([num_test,4096],dtype = np.float32)

    H_B = np.sign(PB.cpu().numpy())
    # knn = neighbors.KNeighborsClassifier()
    # Label = train_labels.numpy()
    # knn.fit(H_B,Label)

  
    clf = svm.SVC()
    
    clf.fit(H_B, train_labels.numpy())

  
    s = 0.0
    for iter, batch in enumerate(test_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]
        uf = torch.randn(pf.size())
        uf = uf.cuda()
        pf = pf.cuda()
        #ff = Variable(ff.cuda(),volatile = True)
        with torch.no_grad():
            fakef = G(pf,uf)
            H_fake, _ = H(fakef)
        H_fake = H_fake.squeeze()
        T[batch_ind.numpy(),:] = torch.sign(H_fake.cpu().data).numpy()
        
        svm_predict = clf.predict(T[batch_ind.numpy(),:].reshape(1,-1))
        # if predict == label:
        #     t+=1
        if svm_predict == label:
            s+=1
        #print(t,s)
    # here map is acc

    #knn_map = (t / float(len(test_loader)))
    svm_map = (s / float(len(test_loader)))
    map = CalcHR.CalcMap(T,H_B,test_labels_onehot.numpy(),train_labels_onehot.numpy())
    print('####################################')
    #print('knn_map:',knn_map)
    print('svm_map:',svm_map)
    print('map:',map)
    #map = round(map,5)
    
    if svm_map > max_map:
        max_map = svm_map
        np.save(str(opt.bit)+"H_B.npy",H_B)
        np.save(str(opt.bit)+'test.npy',T)
        np.save('train_label.npy',train_labels_onehot.numpy())
        np.save('test_label.npy',test_labels_onehot.numpy())
        
