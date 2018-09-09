import torch
from BatchReader import DatasetReader2 as DatasetReader
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
import time


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
parser.add_argument('--lrC', type = float, default = 3e-5, help = "learning rate of c3d" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 30, help = "training epochs")
parser.add_argument('--lamb', type = float, default = 10, help = "lambada")
opt = parser.parse_args()


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
TRAIN_DIR = './TrainSplit1.txt'
TEST_DIR = './TestSplit1.txt'
nclasses = 51


train_data = DatasetReader(TRAIN_DIR)
test_data = DatasetReader(TEST_DIR)

num_train, num_test = len(train_data) , len(test_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 1)
test_loader = DataLoader(test_data,batch_size = opt.batch_size, shuffle = False, num_workers = 1)


train_labels = LoadLabel(TRAIN_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
Y = train_labels_onehot


c3d = networks.C3D()
c3d_dict = c3d.state_dict()
pretrained_dict = torch.load('./c3d.pickle')
pretrained_dict = {k : v for k, v in pretrained_dict.items() if  k in c3d_dict}
c3d_dict.update(pretrained_dict)
c3d.load_state_dict(c3d_dict)
c3d.fc8 = nn.Linear(4096,51)

# c3d = networks.C3D()
# c3d.fc8 = nn.Linear(4096,51)
# #c3d.load_state_dict(torch.load('./hmdb_model.pt'))

# c3d_dict = c3d.state_dict()
# pretrained_dict = torch.load('./hmdb_model.pt')
# pretrained_dict = {k[15:] : v for k, v in pretrained_dict.items() if  k[15:] in c3d_dict}

# c3d_dict.update(pretrained_dict)
# #print(c3d.state_dict().keys())
# c3d.load_state_dict(c3d_dict)
#c3d = torch.load('./checkpoints/3e-05-0.70317models.pt')


c3d = nn.DataParallel(c3d)
c3d.cuda()

#loss
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterion = nn.CrossEntropyLoss().cuda()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()



# Adam optimizer
c3d_optimizer = optim.Adam(c3d.parameters(), lr = opt.lrC, betas = (opt.beta1,opt.beta2))
# training

print("###training start~~~~")

acc_file = open('acc.record','a')
acc_ = 0.0
itr = 0
file = open( str(opt.lrC) + '.log','a')

scheduler = lr_scheduler.StepLR(c3d_optimizer, step_size=20, gamma=0.1)
for epoch in range(opt.train_epoch):
    #c3d.train()
    running_loss = 0.0
    running_corrects = 0.0
    scheduler.step()
    s_time = time.time()
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        c3d.train()
        frame_clip = batch[0]
        #pf = batch[1]
        label_ = batch[1]
        batch_ind = batch[2]

        # train the c3d net
        frame_clip = Variable(frame_clip.cuda())
        label_onehot = EncodingOnehot(label_, nclasses)
        #print(label)
        label_onehot = Variable(label_onehot.cuda())
        label_onehot = torch.unsqueeze(label_onehot,1)
        label = label_.view(-1)
        label = Variable(label.cuda())
        c3d_optimizer.zero_grad()
        logits,_ = c3d(frame_clip)

        _,  preds = torch.max(logits.data, 1)

        loss = criterion(logits, label)
        loss.backward()
        c3d_optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == label.data)
        itr +=1


    epoch_loss = running_loss / ((num_train+0.0)/opt.batch_size)
    running_corrects = running_corrects.cpu().numpy()
    epoch_acc = (running_corrects+0.0) / (num_train+0.0)

    epoch_loss = round(epoch_loss,5)
    epoch_acc = round(epoch_acc,5)

    print('{}, {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, 'train', epoch_loss, epoch_acc))

    #s_time = time.time()
    c3d.eval()
    running_loss = 0.0
    running_corrects = 0.0
    for batch in test_loader:
        frame_clip = batch[0]
        #pf = batch[1]
        label_ = batch[1]
        batch_ind = batch[2]

        label = label_.view(-1)
        frame_clip = Variable(frame_clip.cuda())
        label = Variable(label.cuda())
        with torch.no_grad():
            outputs,_ = c3d(frame_clip)
        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, label)

        # backward + optimize only if in training phase
        #loss.backward()
        #optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(preds == label.data)

    running_corrects = running_corrects.cpu().numpy()
    epoch_loss = running_loss / (num_test/opt.batch_size)
    epoch_acc = running_corrects / (num_test+0.0)
    epoch_loss = round(epoch_loss,5)
    epoch_acc = round(epoch_acc,5)
    e_time = time.time()
    print('{}, {} Loss: {:.4f} Acc: {:.4f} Time:{:.4f}'.format(epoch, 'test', epoch_loss, epoch_acc,(e_time-s_time)))
    acc_file.write('test '+str(epoch_loss)+' '+str(epoch_acc)+ str(opt.lrC) +'\n')
    if epoch_acc > acc_:
        acc_ = epoch_acc
        torch.save(c3d,'./checkpoints/'+str(opt.lrC)+'-'+str(acc_)+'models.pt')
