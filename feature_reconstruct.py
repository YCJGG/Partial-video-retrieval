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
import time


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
parser.add_argument('--bit', type = int, default= 64 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 3e-5, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 1e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 2.5e-3, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 50, help = "training epochs")
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
D = networks.Discriminator(opt.d_input_size,opt.d_hidden_size,opt.d_output_size)
H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)
# print(G)
# print(D)
# print(H)
G.cuda()
D.cuda()
H.cuda()


#loss
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()



# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lrD, betas = (opt.beta1,opt.beta2))
H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

#H_optimizer = optim.SGD(H.parameters(), lr = opt.lrH, momentum = 0.9)
# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))
H_ = torch.zeros(num_train,opt.bit)
FF = torch.zeros(num_train,4096)
PF = torch.zeros(num_train,4096)
max_map = 0
itr = 0
file = open(str(opt.lrG)+'_' + str(opt.lrD)+'_' + str(opt.lrH)+'_' + str(opt.bit) + '.log','a')

scheduler = lr_scheduler.StepLR(H_optimizer, step_size=15, gamma=0.1)

scheduler1 = lr_scheduler.StepLR(G_optimizer, step_size=20, gamma=0.1)
for epoch in range(opt.train_epoch):
    # adjust the lr

    # H_optimizer.param_groups[0]['lr'] = opt.lrH*((epoch//150))
    # if epoch > 150:
    #      G_optimizer.param_groups[0]['lr'] = 0
    #      D_optimizer.param_groups[0]['lr'] = 0
    # E step
    #scheduler.step()
    #scheduler1.step()
    if epoch < 5:
        temp1 = Y.t().mm(Y) +1*torch.eye(nclasses)
        temp1 = temp1.inverse()
        temp1 = temp1.mm(Y.t())
        E = temp1.mm(B)
    #print(D)
    # B step
    B = torch.sign(Y.mm(E) + 1e-5 * H_)

    G.train()
    D.train()
    H.train()
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]

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
        
        fakef  = G(pf, uf)
     
        a = fakef.data.squeeze()
        PF[batch_ind,:] = a.cpu()
        FF[batch_ind,:] = ff.data.squeeze().cpu()

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
        #fake_f_cat = torch.cat((fakef,label),2)
        
        # bs * 2 * 4096
        pred_fake = D.forward(fakef.detach())
        # bs * 2 * 64
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        #real_ab = torch.cat((pf,label),2)
        # bs * 2 * 4096
        pred_real = D.forward(ff)
        # bs * 2 * 64
        loss_d_real = criterionGAN(pred_real, True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        D_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################

        G_optimizer.zero_grad()
         # First, G(A) should fake the discriminator
        #fake_ab = torch.cat((fakef,label),2)

        pred_fake = D.forward(fakef)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb * 5

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()
        G_optimizer.step()

      

        print("===> Epoch[{}]({}/{}): Loss_D: {:.7f} Loss_G: {:.4f} ".format(
            epoch, itr, len(train_loader)*opt.batch_size, loss_d.data[0], loss_g.data[0]))
        itr+=1

    # test per epoch
    # if (epoch+1)%5 == 0:
    #     G.eval()

    
    #     PF_ = np.zeros([num_test,4096],dtype = np.float32)
    #     FF_ = np.zeros([num_test,4096],dtype = np.float32)


    #     # H_B = np.sign(H_.cpu().numpy())
    #     # knn = neighbors.KNeighborsClassifier()
    #     # Label = train_labels.numpy()
    #     # knn.fit(H_B,Label)

    #     #FF_B = FF.cpu().numpy()
    #     PF_B = PF.cpu().numpy()
    # #clf1 = svm.SVC()
    #     clf2 = svm.SVC()
    #     stime = time.time()
    #     #clf1.fit(FF_B, train_labels.numpy())
    #     clf2.fit(PF_B, train_labels.numpy())
    #     etime = time.time()
        
    #     t = 0.0
    #     s = 0.0
    #     for iter, batch in enumerate(test_loader, 0):
    #         ff = batch[0]
    #         pf = batch[1]
    #         label = batch[2]
    #         batch_ind = batch[3]
    #         pf = pf.unsqueeze(1)
    #         uf = torch.randn(pf.size())

    #         uf = uf / uf.pow(2).sum(dim=2).unsqueeze(1)
            
    #         uf = uf.cuda()
    #         ff = ff.cuda()
    #         pf = pf.cuda()
    #         #ff = Variable(ff.cuda(),volatile = True)
    #         with torch.no_grad():
    #             fakef  = G(pf, uf)
    #         PF_B[batch_ind.numpy(),:] = fakef.cpu().data.numpy()
    #         #FF_B[batch_ind.numpy(),:] = ff.cpu().data.numpy()

    #         #predict = knn.predict(T[batch_ind.numpy(),:].reshape(1,-1))
            
    #         #svm_predict1 = clf1.predict(FF_B[batch_ind.numpy(),:].reshape(1,-1))
    #         svm_predict2 = clf2.predict(PF_B[batch_ind.numpy(),:].reshape(1,-1))

    #         # if  svm_predict1 == label:
    #         #     s+=1
    #         if svm_predict2 == label:
    #             t+=1
    #         #print(t,s)
    #     # here map is acc

    #     pfsvm_map = (t / float(len(test_loader)))
    #     #ffsvm_map = (s / float(len(test_loader)))
    #     #map = CalcHR.CalcMap(T,H_B,test_labels_onehot.numpy(),train_labels_onehot.numpy())
    #     print('####################################')
    #     print('pf_map:',pfsvm_map)
    #     #print('ff_map:', ffsvm_map)
    #     #map = round(map,5)
        
    #     if pfsvm_map > max_map:
    #         max_map = pfsvm_map
            # np.save(str(opt.bit)+"H_B.npy",H_B)
            # np.save(str(opt.bit)+'test.npy',T)
            # np.save('train_label.npy',train_labels_onehot.numpy())
            # np.save('test_label.npy',test_labels_onehot.numpy())
torch.save(G,'./G2#_models.pt')







