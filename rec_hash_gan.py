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
def EncodingOnehot1(target, nclasses):
    target_onehot = torch.LongTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

#dataloader
TRAIN_DIR = 'feature_hmdb_TrainSplit1.list'
TEST_DIR = 'feature_hmdb_TestSplit1.list'
ALL_DIR = 'hmdb_ALL_Split1.list'
nclasses = 51


train_data = DatasetProcessing(TRAIN_DIR)
test_data = DatasetProcessing(TEST_DIR)
all_data = DatasetProcessing(ALL_DIR)

num_train, num_test, num_all  = len(train_data) , len(test_data), len(all_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data,batch_size = opt.batch_size, shuffle = False, num_workers = 1)
all_loader = DataLoader(all_data, batch_size = opt.batch_size, shuffle = False, num_workers = 1)

train_labels = LoadLabel(TRAIN_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
all_labels = LoadLabel(ALL_DIR)

Y = train_labels_onehot


G = networks.Generator1(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
D = networks.Discriminator1(opt.d_input_size,opt.d_hidden_size,nclasses, opt.d_output_size,opt.bit)

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
aux_criterion = nn.NLLLoss().cuda()


# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))

params = [
    {'params':D.map1.parameters(), "lr":1e-3},
    {'params':D.map2.parameters(), "lr":3e-3},
    {'params':D.map3.parameters(), "lr":1e-4},
]

D_optimizer = optim.Adam(params, lr = opt.lrD, betas = (opt.beta1,opt.beta2))

# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))

H_ = torch.zeros(num_train,opt.bit)
F_ = torch.zeros(num_train,4096)
max_map = 0
itr = 0
#file = open(str(opt.lrG)+'_' + str(opt.lrD)+'_' + str(opt.lrH)+'_' + str(opt.bit) + '.log','a')

scheduler = lr_scheduler.StepLR(D_optimizer, step_size=100, gamma=0.1)

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
        ori_label = batch[2]
        batch_ind = batch[3]

        ll = ori_label

        ff, pf = Variable(ff.cuda()), Variable(pf.cuda())
        label = EncodingOnehot(ori_label, nclasses)
        aux_label = EncodingOnehot1(ori_label, nclasses)

        aux_label = Variable(aux_label.cuda())
        label = Variable(label.cuda())


        # generate partial feature to full feature

        pf = pf.unsqueeze(1)

  

        fakef  = G(pf)

        a = fakef.data.squeeze()
        F_[batch_ind,:] = a.cpu()
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
 
        pred_fake, H_fake,aux_output = D.forward(fakef.detach())

        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real

        aux_errD_fake = aux_criterion( aux_output, ori_label.squeeze().cuda().view(-1))



        pred_real,H_real,aux_output = D.forward(ff)


        loss_d_real = criterionGAN(pred_real, True)

        # aux_dis
        aux_errD_real = aux_criterion(aux_output, ori_label.squeeze().cuda().view(-1))

        temp = torch.zeros(H_real.data.size())

        for i , ind in enumerate(batch_ind):
            temp[i, :] = B[ind, :]
            H_[ind, :] =H_real.data[i]

        temp = Variable(temp.cuda())
        regterm1 = (temp - H_fake).pow(2).sum()
        regterm2 = (temp - H_real).pow(2).sum()
        regterm3 = (H_real - H_fake).pow(2).sum()



        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5  + (regterm1 + regterm2 + regterm3)/pred_real.size()[0] + (aux_errD_real+aux_errD_fake) * 0.5
        loss_d.backward()

        D_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################

        G_optimizer.zero_grad()
         # First, G(A) should fake the discriminator
        #fake_ab = torch.cat((fakef,label),2)

        pred_fake,_ ,aux_output= D.forward(fakef)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        aux_errG = aux_criterion(aux_output,ori_label.squeeze().cuda().view(-1))

        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb * 2

        loss_g = loss_g_gan + loss_g_l1 + aux_errG

        loss_g.backward()
        G_optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.7f} Loss_G: {:.4f}".format(
            epoch, itr, len(train_loader)*opt.batch_size, loss_d.data[0], loss_g.data[0]))
        itr+=1

    # test per epoch
    G.eval()
    D.eval()
    T = np.zeros([num_test,opt.bit],dtype = np.float32)
    FF = np.zeros([num_test,4096],dtype = np.float32)
    H_B = np.zeros([num_all,opt.bit],dtype = np.float32)
    H_B = np.sign(H_.cpu().numpy())


    H_BB = np.zeros([num_train//10 + num_test,opt.bit],dtype = np.float32)
    all_label = np.zeros(num_train//10 + num_test)
    if (epoch+1)%1 == 0:
        clf = svm.SVC()
        for  i in range(num_train):
            if i%10 == 0:
                H_BB[i//10,:] = H_B[i,:]
                all_label[i//10] = train_labels.numpy()[i]
    # for iteration, batch in enumerate(all_loader, 0):
    #     ff = batch[0]
    #     pf = batch[1]
    #     label = batch[2]
    #     batch_ind = batch[3]

    #     ll = label

    #     ff, pf = Variable(ff.cuda()), Variable(pf.cuda())
    #     with torch.no_grad():
    #         _, H_real,_ = D.forward(ff)
    #     H_real = H_real.squeeze()
    #     H_B[batch_ind.numpy(),:] = np.sign(H_real.cpu().numpy())

    # knn = neighbors.KNeighborsClassifier()
    # Label = train_labels.numpy()
    # knn.fit(H_B,Label)

    # F_B = F_.cpu().numpy()
    #clf = svm.SVC()
    # stime = time.time()

    # etime = time.time()
    if (epoch+1)%1 == 0:
        t = 0.0
        s = 0.0
        correct = 0.0
        for iter, batch in enumerate(test_loader, 0):
            ff = batch[0]
            pf = batch[1]
            label = batch[2]
            batch_ind = batch[3]

            pf = pf.unsqueeze(1)


            pf = pf.cuda()
            label = label.cuda()
            ff = ff.cuda()
            #ff = Variable(ff.cuda(),volatile = True)
            with torch.no_grad():
                fakef  = G(pf)
                _,H_fake,output = D(fakef)
                _,H_real,_ = D(ff)

            # pred = output.data.max(1, keepdim=True)[1]
            # #print(pred)
            # correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            # #print(correct)
            H_fake = H_fake.squeeze()
            H_real = H_real.squeeze()
            T[batch_ind.numpy(),:] = torch.sign(H_fake.cpu().data).numpy()
            H_BB[num_train//10 + batch_ind.numpy(),:] = torch.sign(H_real.cpu().data).numpy()
            all_label[num_train//10 + batch_ind.numpy()] = label.cpu().data.numpy().reshape(-1)
        #     FF[batch_ind.numpy(),:] = fakef.cpu().data.numpy()
        clf.fit(H_BB, all_label)
        svm_predict = clf.predict(T.reshape(-1,opt.bit))

        # # here map is acc

        s = sum(svm_predict.reshape(-1) == test_labels.numpy().reshape(-1))

        # knn_map = (t / float(len(test_loader)))
        svm_map = (s / float(num_test))

        #acc = float(correct) / float(num_test)

        #H_B_ = np.sign(H_.cpu().numpy())
        #map = CalcHR.CalcMap(T,H_B_,test_labels_onehot.numpy(),train_labels_onehot.numpy())
        print('####################################')
    # #    print('knn_map:',knn_map)
        #print('map:',map)
        print('acc:',svm_map)
        print('####################################')
        #map = round(map,5)

        if svm_map > max_map:
            max_map = svm_map
            # np.save(str(opt.bit)+"H_B.npy",H_B)
            # np.save(str(opt.bit)+'test.npy',T)
            # np.save('train_label.npy',train_labels_onehot.numpy())
            # np.save('test_label.npy',test_labels_onehot.numpy())
            torch.save(G,'./'+str(opt.bit)+TRAIN_DIR+'G3_models.pt')
            #torch.save(H,'./H3_models.pt')







