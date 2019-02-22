import torch
from BatchReader import DatasetReader_fullvid, DatasetReader_random_partialvid
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import networks
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os,glob
import pdb

torch.backends.cudnn.benchmark = True

#dataloader
#ALL_DIR = '../trainVidID.txt'
ALL_DIR = glob.glob('../test_frames/*')
ALL_DIR = [x.split('/')[-1] for x in ALL_DIR]
nclasses = 51

c3d = networks.C3D()
c3d_dict = c3d.state_dict()
c3d_pretrain_dict = torch.load('./c3d.pickle')
c3d_pretrain_dict = {k:v for k,v in c3d_pretrain_dict.items() if k in c3d_dict}
c3d_dict.update(c3d_pretrain_dict)
c3d.load_state_dict(c3d_dict)
c3d.fc8 = nn.Linear(4096,51)
c3d.cuda()
c3d.eval()

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm

print("###extracting start~~~~")

"""
if not os.path.exists('../full_train_features/'):
	os.mkdir('../full_train_features/')
#ALL_DIR = file(ALL_DIR).readlines()
for idx, line in enumerate(ALL_DIR):
	print 'Processing {} {}/{}'.format(line.strip(), idx, len(ALL_DIR))
	if os.path.isfile('../full_train_features/'+line.strip()+'.npy') == True:
		continue
	vid_dir = '../train_frames/' + line.strip()
	ff_data = DatasetReader_fullvid(vid_dir)
	ff_dataloader = DataLoader(ff_data, batch_size = 1, shuffle = False, num_workers = 2)
	ff_all = []
	for frame_clip in ff_dataloader:
		frame_clip = Variable(frame_clip).cuda()
		with torch.no_grad():
			#pdb.set_trace()
			_,ff = c3d(frame_clip)
			ff_all.append(ff.cpu().numpy())
	ff = normalize( np.mean(ff_all, axis=0) )
	#print ff
	np.save('../full_train_features/'+line.strip()+'.npy', ff)

"""

if not os.path.exists('../partial_test_features/'):
	os.mkdir('../partial_test_features/')
partial_info = file('test_partial_info.txt','w')
for idx, line in enumerate(ALL_DIR):
	print 'Processing {} {}/{}'.format(line.strip(), idx, len(ALL_DIR))
	for i in range(10):
		if os.path.isfile('../partial_test_features/'+line.strip()+'_'+str(i)+'.npy') == True:
			continue
		vid_dir = '../test_frames/' + line.strip()
		pf_data = DatasetReader_random_partialvid(vid_dir)
		pf_dataloader = DataLoader(pf_data, batch_size = 1, shuffle = False, num_workers = 2)

		start, end, ratio = pf_data.get_start_end_frame_id()
		partial_info.write( line.strip() + '\t' + str(i) + '\t' + str(start) + '\t' + str(end) + '\t' + '{:0.3}'.format(ratio) + '\n' )

		pf_all = []
		for frame_clip in pf_dataloader:
			frame_clip = Variable(frame_clip).cuda()
			with torch.no_grad():
				#pdb.set_trace()
				_,pf = c3d(frame_clip)
				pf_all.append(pf.cpu().numpy())
		pf = normalize( np.mean(pf_all, axis=0) )
		#print pf
		np.save('../partial_test_features/'+line.strip()+'_'+str(i)+'.npy', pf)
partial_info.close()
