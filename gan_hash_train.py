import torch
from BatchReader import DatasetProcessing, DatasetReader
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


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default= 1, help = "batch size")
parser.add_argument('--g_input_size', type = int, default= 4096+51, help = "input size of generator")
parser.add_argument('--g_hidden_size', type = int, default= 4096*2, help = "hidden size of generator")
parser.add_argument('--g_output_size', type = int, default= 4096, help = "output size of generator")
parser.add_argument('--d_input_size', type = int, default= 4096+51, help = "input size of discriminator")
parser.add_argument('--d_hidden_size', type = int, default= 1024, help = "hidden size of discriminator")
parser.add_argument('--d_output_size', type = int, default= 64 , help = "output size of discriminator")
parser.add_argument('--h_input_size', type = int, default= 4096, help = "input size of Hashnet")
parser.add_argument('--h_hidden_size', type = int, default= 1024, help = "hidden size of Hashnet")
parser.add_argument('--bit', type = int, default= 64 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 2e-5, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 2e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 1e-4, help = "learning rate of Hashnet" )
parser.add_argument('--lrC', type = float, default = 1e-4, help = "learning rate of c3d" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 300, help = "training epochs")
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


#dataloader
TRAIN_DIR = './list/train.list'
TEST_DIR = './list/test.list'
nclasses = 51


train_data = DatasetReader(TRAIN_DIR)
test_data = DatasetReader(TEST_DIR)

num_train, num_test = len(train_data) , len(test_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data,batch_size = opt.batch_size, shuffle = False, num_workers = 4)
# for i in train_loader:
#     print(i[0].size())

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
c3d = networks.C3D()
c3d_dict = c3d.state_dict()
pretrained_dict = torch.load('./c3d.pickle')
pretrained_dict = {k : v for k, v in pretrained_dict.items() if  k in c3d_dict}
c3d_dict.update(pretrained_dict)
c3d.load_state_dict(c3d_dict)
c3d.fc8 = nn.Linear(4096,51)

G.cuda()
D.cuda()
H.cuda()
c3d.cuda()

#loss 
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterion = nn.CrossEntropyLoss().cuda()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()



# Adam optimizer
c3d_optimizer = optim.Adam(c3d.parameters(), lr = opt.lrC, betas = (opt.beta1,opt.beta2))
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lrD, betas = (opt.beta1,opt.beta2))
H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))
H_ = torch.zeros(num_train,opt.bit)

max_map = 0
itr = 0
file = open(str(opt.lrG)+'_' + str(opt.lrD)+'_' + str(opt.lrH)+'_' + str(opt.bit) + '.log','a')
for epoch in range(opt.train_epoch):
    # adjust the lr 
    #H_optimizer.param_groups[0]['lr'] = opt.lrH*(0.1**(epoch//70))

    # E step
    temp1 = Y.t().mm(Y) +torch.eye(nclasses)
    temp1 = temp1.inverse()
    temp1 = temp1.mm(Y.t())
    E = temp1.mm(B)
    #print(D)
    # B step 
    B = torch.sign(Y.mm(E) + 1e-5 * H_)
    c3d.train()
    G.train()
    D.train()
    H.train()
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        c3d.train()
        frame_clip = batch[0]
        #pf = batch[1]
        label_ = batch[1]
        batch_ind = batch[2]
        frames = batch[3]
        frames =  Variable(frames.cuda())
        # train the c3d net
        frame_clip = Variable(frame_clip.cuda())
        label_onehot = EncodingOnehot(label_, nclasses)
        #print(label)
        label_onehot = Variable(label_onehot.cuda())
        label_onehot = torch.unsqueeze(label_onehot,1)
        label = label_.view(-1)
        label = Variable(label.cuda())
        #print(label.view(-1).size())
        c3d_optimizer.zero_grad()
        #print(frame_clip.size())
        logits,_ = c3d(frame_clip)
        
        #print(logits.size())
        #_, preds = torch.max(logits.data, 1)
        #print(type(preds))
        #print(type(label))
        #print(preds.size())
        #print(label.size())
        class_loss = criterion(logits, label)
        class_loss.backward()
        c3d_optimizer.step()
        #print(preds)

        c3d.eval()
        with torch.no_grad():
            frames = torch.squeeze(frames)
            if len(frames.size())<=4:
                frames = torch.unsqueeze(frames,0)
            _,ff = c3d(frames)
            ff = torch.mean(ff, dim = 0)
            ff = ff.view(1,1,-1)
            #ff = torch.norm(ff,p=2)
            #pf = torch.norm(c3d(frame_clip)[1],p=2)
            _,pf = c3d(frame_clip)
            #ff = ff.view(1,-1)
            #pf = pf.view(1,-1)
            #print(pf.size())
        #ff, pf = Variable(ff.cuda()), Variable(pf.cuda())
        
        # generate partial feature to full feature
        #print(pf.size())
        #print(label.size())
        #label = label.float().view(-1,1)
        #print(pf)
        pf = torch.unsqueeze(pf,1)
        pf_cat = torch.cat((pf,label_onehot),2)
        #print(pf_cat)

        fakef  = G(pf_cat)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
        fake_f_cat = torch.cat((fakef,label_onehot),2)
        # bs * 2 * 4096 
        pred_fake = D.forward(fake_f_cat.detach())
        # bs * 2 * 64
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        real_ab = torch.cat((pf,label_onehot),2)
        # bs * 2 * 4096
        pred_real = D.forward(real_ab)
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
        fake_ab = torch.cat((fakef,label_onehot),2)
        
        pred_fake = D.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()
        G_optimizer.step()

        H_optimizer.zero_grad()
        cat = torch.zeros([pf.size()[0],1,nclasses])
        cat = Variable(cat.cuda())
        pf_cat_ = torch.cat((pf,cat),2)
        fakef = G(pf_cat_)
        H_fake = H(fakef)
        H_real = H(ff)
        temp = torch.zeros(H_real.data.size())
        for i , ind in enumerate(batch_ind):
            temp[i, :] = B[ind, :]
            H_[ind, :] = H_real.data[i]
        temp = Variable(temp.cuda())
        regterm1 = (temp - H_fake).pow(2).sum()
        regterm2 = (temp - H_real).pow(2).sum()
        regterm3 = (H_real - H_fake).pow(2).sum()
        
        H_loss = (regterm1 +regterm2 + regterm3)/opt.batch_size
        
        H_loss.backward()
        H_optimizer.step()

        
        print("===> Epoch[{}]({}/{}): Loss_D: {:.7f} Loss_G: {:.4f} Loss_H: {:.4f}  Loss_C: {:.4f}".format(
            epoch, itr, len(train_loader)*opt.batch_size, loss_d.data[0], loss_g.data[0],H_loss.data[0],class_loss.data[0]))
        itr+=1

    # test per epoch
    G.eval()
    H.eval()
    c3d.eval()
    T = np.zeros([num_test,opt.bit],dtype = np.float32)
    for iter, batch in enumerate(test_loader, 0):
        frame_clip = batch[0]
        #pf = batch[1]
        label_ = batch[1]
        batch_ind = batch[2]
        frames = batch[3]
        #frames =  Variable(frames.cuda())
        # train the c3d net
        frame_clip = Variable(frame_clip.cuda())
        label_onehot = EncodingOnehot(label_, nclasses)
        #print(label)
        label_onehot = Variable(label_onehot.cuda())
        label_onehot = torch.unsqueeze(label_onehot,1)
        
        with torch.no_grad():
            _,pf = c3d(frame_clip)
            pf = torch.unsqueeze(pf,1)
            cat = torch.zeros([pf.size()[0],1,nclasses])
            cat = Variable(cat.cuda())
            pf = Variable(pf.cuda())
            pf_cat_ = torch.cat((pf,cat),2)
        #ff = Variable(ff.cuda(),volatile = True)
        
            fakef = G(pf_cat_)
            H_fake = H(fakef)
        H_fake = H_fake.squeeze()
        T[batch_ind.numpy(),:] = torch.sign(H_fake.cpu().data).numpy()

    # map
    H_B = np.sign(H_.cpu().numpy())
    map = CalcHR.CalcMap(T,H_B,test_labels_onehot.numpy(),train_labels_onehot.numpy())
    print('####################################')
    print('map:',map)
    map = round(map,5)
    file.write(str(map) +'\n')
    print('####################################')
    if map > max_map:
        max_map = map
        np.save(str(opt.bit)+"H_B.npy",H_B)
        np.save(str(opt.bit)+'test.npy',T)
        np.save('train_label.npy',train_labels_onehot.numpy())
        np.save('test_label.npy',test_labels_onehot.numpy())






