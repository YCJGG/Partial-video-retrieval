import torch
from BatchReader import DatasetProcessing2 as DatasetProcessing
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
import scipy.io as scio 


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default= 256, help = "batch size")
parser.add_argument('--g_input_size', type = int, default= 4096, help = "input size of generator")
parser.add_argument('--g_hidden_size', type = int, default= 4096, help = "hidden size of generator")
parser.add_argument('--g_output_size', type = int, default= 4096, help = "output size of generator")
parser.add_argument('--d_input_size', type = int, default= 4096, help = "input size of discriminator")
parser.add_argument('--d_hidden_size', type = int, default= 1024, help = "hidden size of discriminator")
parser.add_argument('--d_output_size', type = int, default= 64 , help = "output size of discriminator")
parser.add_argument('--h_input_size', type = int, default= 4096, help = "input size of Hashnet")
parser.add_argument('--h_hidden_size', type = int, default= 2048, help = "hidden size of Hashnet")
parser.add_argument('--bit', type = int, default= 64 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 3e-5, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 1e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 1e-3, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 150, help = "training epochs")
parser.add_argument('--lamb', type = float, default = 10, help = "lambada")
opt = parser.parse_args()


# load labels
def LoadLabel(filename):
    fp = open(filename,'r')
    labels = [x.strip().split()[2] for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int,labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


#dataloader
TRAIN_DIR = 'feature_hmdb_TrainSplit1.list'
TEST_DIR = 'feature_hmdb_TestSplit1.list'
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


G = networks.Generator(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)
# print(G)
# print(D)
# print(H)

G_dict = G.state_dict()
pretrained_dict = torch.load('./G2#_models.pt')
pretrained_dict = pretrained_dict.state_dict()
pretrained_dict = {k : v for k, v in pretrained_dict.items() if  k in G_dict}
G_dict.update(pretrained_dict)
G.load_state_dict(G_dict)



G.cuda()
H.cuda()




H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH,  betas = (opt.beta1,opt.beta2))

print("###training start~~~~")

# initialize the B and H
#B = torch.sign(torch.randn(num_train, opt.bit))

B = scio.loadmat('B_init.mat')['B']
B = torch.FloatTensor(B)
H_ = torch.zeros(num_train,opt.bit)
F_ = torch.zeros(num_train,4096)

trainF = np.zeros([num_train,4096],dtype = np.float32)
max_map = 0
itr = 0
scheduler = lr_scheduler.StepLR(H_optimizer, step_size=15, gamma=0.1)

temp1 = Y.t().mm(Y) +1*torch.eye(nclasses)
temp1 = temp1.inverse()
temp1 = temp1.mm(Y.t())

for epoch in range(opt.train_epoch):
    # adjust the lr

    # H_optimizer.param_groups[0]['lr'] = opt.lrH*((epoch//150))
    # if epoch > 150:
    #      G_optimizer.param_groups[0]['lr'] = 0
    #      D_optimizer.param_groups[0]['lr'] = 0
    # E step
    #scheduler.step()
    #scheduler1.step()
    

    E = temp1.mm(B)
    #print(D)
    # B step
    B = torch.sign(Y.mm(E) + 1e-5 * H_)
    
    G.eval()
    H.train()
    running_loss = 0.0
    itt = 0
    for iteration, batch in enumerate(train_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]

        trainF[batch_ind,:] = pf.numpy()

        ll = label

        ff, pf = Variable(ff.cuda()), Variable(pf.cuda())

        label = EncodingOnehot(label, nclasses)
        
        label = Variable(label.cuda())
        label = torch.unsqueeze(label,1)
        # generate partial feature to full feature
       
        #pf_cat = torch.cat((pf,label),2)


        pf = pf.unsqueeze(1)

        uf = torch.randn(pf.size())

        uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
        
        uf = uf.cuda()
        
        fakef  = G(pf, uf)


        H_optimizer.zero_grad()
    
        #cat = torch.zeros([pf.size()[0],1,nclasses])
        #cat = Variable(cat.cuda())
        #pf_cat_ = torch.cat((pf,cat),2)
        #fakef = G(pf_cat_)
        #H_fake = H(fakef)

        #F_[batch_ind,:] = fakef.data.cpu().squeeze()

        H_fake = H(fakef)
        
        H_real = H(ff)
        temp = torch.zeros(H_fake.data.size())
        regterm4 = 0.0
        for i , ind in enumerate(batch_ind):
            temp[i, :] = B[ind, :]
            H_[ind, :] = H_real.data[i]
            # k = i
            # while(k == i):
            #     p = np.random.randint(len(batch_ind))
            #     k = p
            # if ll[i] == ll[k]:
            #     regterm4 += ((H_real[i] - H_real[k]).pow(2).sum()+ (H_fake[i] - H_fake[k]).pow(2).sum())
            
        temp = Variable(temp.cuda())
        regterm1 = (temp - H_fake).pow(2).sum()
        regterm2 = (temp - H_real).pow(2).sum()
        regterm3 = (H_real - H_fake).pow(2).sum()


        

        H_loss = (regterm1 +regterm2 + regterm3 )/opt.batch_size

        H_loss.backward()
        H_optimizer.step()
        running_loss += H_loss.item()
        itt+=1
    print(epoch, running_loss/float(itt))
    

    scio.savemat('trainF.mat', {'trainF':trainF})  
    if(epoch+1)%2 == 0:



        G.eval()
        H.eval()
        T = np.zeros([num_test,opt.bit],dtype = np.float32)
        FF = np.zeros([num_test,4096],dtype = np.float32)

        H_B = np.sign(H_.cpu().numpy())
        knn = neighbors.KNeighborsClassifier()
        Label = train_labels.numpy()
        knn.fit(H_B,Label)
    # F_ = F_.numpy()
    
        # clf = svm.SVC()
        
        # clf.fit(H_B, train_labels.numpy())

    
        s = 0.0
        for iter, batch in enumerate(test_loader, 0):
            ff = batch[0]
            pf = batch[1]
            label = batch[2]
            batch_ind = batch[3]
            # cat = torch.zeros([pf.size()[0],1,nclasses])
            # cat = Variable(cat.cuda())
            # pf = Variable(pf.cuda())
            # pf_cat_ = torch.cat((pf,cat),2)
            #ff = Variable(ff.cuda(),volatile = True)
            pf = pf.unsqueeze(1)
            pf = pf.cuda()

            uf = torch.randn(pf.size())

            uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
            
            uf = uf.cuda()


            with torch.no_grad():
                fakef = G(pf, uf)
                H_fake = H(fakef)
            H_fake = H_fake.squeeze()
            T[batch_ind.numpy(),:] = torch.sign(H_fake.cpu().data).numpy()
            #FF[batch_ind.numpy(),:] = fakef.cpu().data.numpy()
        svm_predict = knn.predict(T)
            # if predict == label:
            #     t+=1
        s = sum( svm_predict.reshape(-1) == test_labels.numpy().reshape(-1))
            
            #print(t,s)
        # here map is acc

        #knn_map = (t / float(len(test_loader)))
        print(s)
        svm_map = (s / float(len(test_loader)))
        map = CalcHR.CalcMap(T,H_B,test_labels_onehot.numpy(),train_labels_onehot.numpy())
        print('####################################')
        #print('knn_map:',knn_map)
        print('svm_map:',svm_map)
        print('map',map)
        #map = round(map,5)
        
        if svm_map > max_map:
            max_map = svm_map
            np.save(str(opt.bit)+"H_B.npy",H_B)
            np.save(str(opt.bit)+'test.npy',T)
            np.save('train_label.npy',train_labels_onehot.numpy())
            np.save('test_label.npy',test_labels_onehot.numpy())
            
