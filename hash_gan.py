import torch
from BatchReader import DatasetProcessing2 as  DatasetProcessing
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
import time
import scipy.io as scio

nclasses = 51

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
parser.add_argument('--bit', type = int, default= 128 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 3e-5, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 1e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 1e-3, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 160, help = "training epochs")
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
TRAIN_DIR = 'Retrieval_hmdb_TrainSplit.list'
TEST_DIR = 'Retrieval_hmdb_TestSplit.list'
nclasses = 51


train_data = DatasetProcessing(TRAIN_DIR)
test_data = DatasetProcessing(TEST_DIR)

num_train, num_test = len(train_data) , len(test_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data,batch_size = opt.batch_size, shuffle = False, num_workers = 1)


train_labels = LoadLabel(TRAIN_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
Y = train_labels_onehot


G = networks.Generator1(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
D = networks.Discriminator1(opt.d_input_size,opt.d_hidden_size,opt.bit)
#H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)
# print(G)
# print(D)
# print(H)
G.cuda()
D.cuda()
#H.cuda()


#loss
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()



# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))

params = [
    {'params':D.map1.parameters(), "lr":1e-5},
    {'params':D.map2.parameters(), "lr":1e-3},
    {'params':D.map3.parameters(), "lr":1e-5},
]

D_optimizer = optim.Adam(params, lr = opt.lrD, betas = (opt.beta1,opt.beta2))
#H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

#H_optimizer = optim.SGD(H.parameters(), lr = opt.lrH, momentum = 0.9)
# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))

# B = scio.loadmat('B_init.mat')['B']
# B = torch.FloatTensor(B)

H_ = torch.zeros(num_train,opt.bit)
F_ = torch.zeros(num_train,4096)
max_map = 0
itr = 0
#file = open(str(opt.lrG)+'_' + str(opt.lrD)+'_' + str(opt.lrH)+'_' + str(opt.bit) + '.log','a')

scheduler = lr_scheduler.StepLR(D_optimizer, step_size=120, gamma=0.1)

scheduler1 = lr_scheduler.StepLR(G_optimizer, step_size=20, gamma=0.1)
for epoch in range(opt.train_epoch):
    # adjust the lr

    # H_optimizer.param_groups[0]['lr'] = opt.lrH*((epoch//150))
    # if epoch > 150:
    #      G_optimizer.param_groups[0]['lr'] = 0
    #      D_optimizer.param_groups[0]['lr'] = 0
    # E step
    scheduler.step()
    #scheduler1.step()
    if epoch < 15:
        temp1 = Y.t().mm(Y) +1*torch.eye(nclasses)
        temp1 = temp1.inverse()
        temp1 = temp1.mm(Y.t())
        E = temp1.mm(B)
    #print(D)
    # B step
    B = torch.sign(Y.mm(E) + 1e-5 * H_)

    G.train()
    D.train()
    
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]

        ll = label

        ff, pf = Variable(ff.cuda()), Variable(pf.cuda())
        label = EncodingOnehot(label, nclasses)
        #print(label)
        label = Variable(label.cuda())
        label = torch.unsqueeze(label,1)
        # generate partial feature to full feature
        #print(pf.size())
        #print(label.size())
        #pf_cat = torch.cat((pf,label),2)
        #print(pf_cat)

        pf = pf.unsqueeze(1)

        uf = torch.randn(pf.size())

        uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
        
        uf = uf.cuda()
        
        fakef  = G(pf)
     
        a = fakef.data.squeeze()
        F_[batch_ind,:] = a.cpu()
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
        #fake_f_cat = torch.cat((fakef,label),2)
        
        
        pred_fake,H_fake = D.forward(fakef.detach())
        
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
    
        pred_real,H_real = D.forward(ff)
        
        loss_d_real = criterionGAN(pred_real, True)

        temp = torch.zeros(H_real.data.size())
        
        for i , ind in enumerate(batch_ind):
            temp[i, :] = B[ind, :]
            H_[ind, :] =H_real.data[i]
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



        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5 + (regterm1 + regterm2 + regterm3)/pred_real.size()[0]

        loss_d.backward()

        D_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################

        G_optimizer.zero_grad()
         # First, G(A) should fake the discriminator
        #fake_ab = torch.cat((fakef,label),2)

        pred_fake,_ = D.forward(fakef)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb * 2

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()
        G_optimizer.step()

        # H_optimizer.zero_grad()
        # #cat = torch.zeros([pf.size()[0],1,nclasses])
        # #cat = Variable(cat.cuda())
        # #pf_cat_ = torch.cat((pf,cat),2)

        # uf = torch.randn(pf.size())

        # uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
        
        # uf = uf.cuda()
        
        # fakef  = G(pf)

        # #H_fake = H(fakef)
        # H_fake = H(fakef)
        
        # H_real = H(ff)
        # temp = torch.zeros(H_fake.data.size())
        # regterm4 = 0.0
        # for i , ind in enumerate(batch_ind):
        #     temp[i, :] = B[ind, :]
        #     H_[ind, :] =H_real.data[i]
        #     # k = i
        #     # while(k == i):
        #     #     p = np.random.randint(len(batch_ind))
        #     #     k = p
        #     # if ll[i] == ll[k]:
        #     #     regterm4 += ((H_real[i] - H_real[k]).pow(2).sum()+ (H_fake[i] - H_fake[k]).pow(2).sum())
            
        # temp = Variable(temp.cuda())
        # regterm1 = (temp - H_fake).pow(2).sum()
        # regterm2 = (temp - H_real).pow(2).sum()
        # regterm3 = (H_real - H_fake).pow(2).sum()

        # H_loss = (regterm1 +regterm2 + regterm3 )/H_real.size()[0]
        # #H_loss = (regterm1)/opt.batch_size 
        # H_loss.backward()
        # H_optimizer.step()


        print("===> Epoch[{}]({}/{}): Loss_D: {:.7f} Loss_G: {:.4f}".format(
            epoch, itr, len(train_loader)*opt.batch_size, loss_d.data[0], loss_g.data[0]))
        itr+=1

    # test per epoch
    G.eval()
    D.eval()
    T = np.zeros([num_test,opt.bit],dtype = np.float32)
    FF = np.zeros([num_test,4096],dtype = np.float32)



    H_B = np.sign(H_.cpu().numpy())
    # knn = neighbors.KNeighborsClassifier()
    # Label = train_labels.numpy()
    # knn.fit(H_B,Label)

    # F_B = F_.cpu().numpy()
    # clf = svm.SVC()
    # stime = time.time()
    # clf.fit(F_B, train_labels.numpy())
    # etime = time.time()
    if (epoch+1)%4 == 0:        
        t = 0.0
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


            # pf = pf.unsqueeze(1)
            
            
            # uf = torch.randn(pf.size())

            # uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
            
            # uf = uf.cuda()
            pf = pf.cuda()
            
            #ff = Variable(ff.cuda(),volatile = True)
            with torch.no_grad():
                fakef  = G(pf)
                _,H_fake = D(fakef)
                
            H_fake = H_fake.squeeze()
            T[batch_ind.numpy(),:] = torch.sign(H_fake.cpu().data).numpy()
        #     FF[batch_ind.numpy(),:] = fakef.cpu().data.numpy()
        #     predict = knn.predict(T[batch_ind.numpy(),:].reshape(1,-1))
        #     #svm_predict = clf.predict(FF[batch_ind.numpy(),:].reshape(1,-1))
        #     if predict == label:
        #         t+=1
        #     # if svm_predict == label:
        #     #     s+=1
        #     #print(t,s)
        # # here map is acc

        # knn_map = (t / float(len(test_loader)))
        #svm_map = (s / float(len(test_loader)))
        map = CalcHR.CalcMap(T,H_B,test_labels_onehot.numpy(),train_labels_onehot.numpy())
        print('####################################')
    #    print('knn_map:',knn_map)
        print('map:',map)
        #print('svm_map:',svm_map)
        #map = round(map,5)
        
        # if map > max_map:
        #     max_map = map
        #     np.save(str(opt.bit)+"H_B.npy",H_B)
        #     np.save(str(opt.bit)+'test.npy',T)
        #     np.save('train_label.npy',train_labels_onehot.numpy())
        #     np.save('test_label.npy',test_labels_onehot.numpy())
        #     torch.save(G,'./G3_models.pt')
        #     torch.save(H,'./H3_models.pt')







