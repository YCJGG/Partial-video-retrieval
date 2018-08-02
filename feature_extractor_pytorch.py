import torch
from BatchReader import DatasetReader as DatasetReader
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
import os


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
parser.add_argument('--train_epoch', type = int, default = 20, help = "training epochs")
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
TRAIN_DIR = './list/train.list'
TEST_DIR = './list/test.list'
ALL_DIR = './list/image.list'
nclasses = 51
if not os.path.exists('./partial_features/'):
    os.mkdir('./partial_features/')
if not os.path.exists('./full_features/'):
    os.mkdir('./full_features/')
# train_data = DatasetReader(TRAIN_DIR)
# test_data = DatasetReader(TEST_DIR)
all_data = DatasetReader(ALL_DIR)

#num_train, num_test = len(train_data) , len(test_data)
num_data  = len(all_data)

#train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
all_loader = DataLoader(all_data,batch_size = opt.batch_size, shuffle = False, num_workers = 4)


all_labels = LoadLabel(ALL_DIR)
all_labels_onehot = EncodingOnehot(all_labels, nclasses)
#test_labels = LoadLabel(TEST_DIR)
#test_labels_onehot = EncodingOnehot(test_labels, nclasses)
#Y = train_labels_onehot


c3d = networks.C3D()
c3d.fc8 = nn.Linear(4096,51)
# #c3d.load_state_dict(torch.load('./hmdb_model.pt'))

c3d_dict = c3d.state_dict()
# pretrained_dict = torch.load('./hmdb_model.pt')
# pretrained_dict = {k[15:] : v for k, v in pretrained_dict.items() if  k[15:] in c3d_dict}

# c3d_dict.update(pretrained_dict)
# #print(c3d.state_dict().keys())
# c3d.load_state_dict(c3d_dict)
# # c3d.fc8 = nn.Linear(4096,51)
c3d_pretrain = torch.load('./checkpoints/3e-05-0.70317models.pt')
c3d_pretrain_dict = c3d_pretrain.state_dict()
c3d_pretrain_dict = {k[7:]:v for k,v in c3d_pretrain_dict.items() if k[7:] in c3d_dict}
c3d.load_state_dict(c3d_pretrain_dict)
#print(c3d.state_dict()['fc8.bias'])
#c3d = nn.DataParallel(c3d)
c3d.cuda()
#torch.save(c3d.state_dict(),'./hmdb_model.pt')
#loss 
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterion = nn.CrossEntropyLoss().cuda()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
# Adam optimizer
#c3d_optimizer = optim.Adam(c3d.parameters(), lr = opt.lrC, betas = (opt.beta1,opt.beta2))
# training

print("###extracting start~~~~")


#s_time = time.time()
c3d.eval()
file = open('./features.list','a')
for batch in all_loader:
    frame_clip = batch[0]
    #pf = batch[1]
    label_ = batch[1]
    batch_ind = batch[2]
    frames = batch[3]
    name = batch[4]
    label = label_.view(-1)
    frame_clip = Variable(frame_clip.cuda())
    label = Variable(label.cuda())
    frames = Variable(frames.cuda())
    label_onehot = EncodingOnehot(label_, nclasses)
    #print(label)
    label_onehot = Variable(label_onehot.cuda())
    label_onehot = torch.unsqueeze(label_onehot,1)
    with torch.no_grad():
        _,pf = c3d(frame_clip)
        frames = torch.squeeze(frames)

    if len(frames.size())<=4:
        frames = torch.unsqueeze(frames,0)
    with torch.no_grad():
        _,ff = c3d(frames)
        ff = torch.mean(ff, dim = 0)
            #ff = ff.view(1,1,-1)
    ff = ff.view(1,-1)
    pf = pf.cpu().numpy()
    ff = ff.cpu().numpy()
    label = label_.cpu().numpy()

    partial_feature = normalize(pf) 
    full_feature = normalize(ff)
    path = name[0].split('//')[1]
    cate_name = path.split('/')[0]
    file_name =  path.split('/')[1]
    if not os.path.exists('./partial_features/'+cate_name):
        os.mkdir('./partial_features/'+cate_name)
    if not os.path.exists('./full_features/'+cate_name):
        os.mkdir('./full_features/'+cate_name)
    np.save('./partial_features/'+path+'.npy', partial_feature)
    np.save('./full_features/'+path+'.npy', full_feature)
    file.write('./full_features/'+path+'.npy'+' '+'./partial_features/'+path+'.npy'+' '+str(label)+'\n')

    #print(partial_feature)
    #print(full_feature.shape)
    


    
