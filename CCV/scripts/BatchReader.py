import numpy as np
import torch.utils.data as data
import torch
import os, glob
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

class DatasetReader_fullvid(data.Dataset):
    def __init__(self, video_path):
        self.video_path = video_path

        self.num_frames_per_clip = 16
        self.crop_size = 112
        self.np_mean = np.load('crop_mean.npy').reshape([self.num_frames_per_clip, self.crop_size, self.crop_size, 3])
    

    def __getitem__(self,index):
        clip_frames = []
        s_index = index * 8
        for idx in range(s_index, s_index + self.num_frames_per_clip):
	    image_name = self.video_path + '/{:05}.jpg'.format(idx)
	    #print(image_name)
	    img = Image.open(image_name)
	    img_data = np.array(img)
	    crop_x = int((img_data.shape[0] - self.crop_size)/2)
	    crop_y = int((img_data.shape[1] - self.crop_size)/2)
	    img_data = img_data[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size,:] - self.np_mean[idx%8]
	    clip_frames.append(img_data)

        clip_frames = torch.from_numpy( np.array(clip_frames).astype(np.float32).transpose(3,0,1,2) )

        return clip_frames

    def __len__(self):
	num_frames = len(glob.glob(self.video_path + '/*.jpg'))
        return int((num_frames-8)/8)


class DatasetReader_random_partialvid(data.Dataset):
    def __init__(self, video_path):
	# video path is a folder path that contains all video frames
        self.video_path = video_path

        self.num_frames_per_clip = 16
        self.crop_size = 112
        self.np_mean = np.load('crop_mean.npy').reshape([self.num_frames_per_clip, self.crop_size, self.crop_size, 3])

	# randomly select a partial clip
	self.num_frames = len(glob.glob(self.video_path + '/*.jpg'))
	for try_ in range(10):
	    self.random_clip_len = np.random.uniform(low=0.105, high=0.3) * self.num_frames
	    self.random_clip_len = int(self.random_clip_len / 8) * 8
	    if self.random_clip_len >= 16:
		break
	if self.random_clip_len < 16:
	    	self.random_clip_len = 16
	self.start = np.random.randint(low=0, high=self.num_frames-self.random_clip_len)
	self.end = self.start + self.random_clip_len
    

    # get a clip (16 frames)
    def __getitem__(self,index):
        clip_frames = []
        s_index = index * 8 + self.start
        for idx in range(s_index, s_index + self.num_frames_per_clip):
	    image_name = self.video_path + '/{:05}.jpg'.format(idx)
	    #print(image_name)
	    img = Image.open(image_name)
	    img_data = np.array(img)
	    crop_x = int((img_data.shape[0] - self.crop_size)/2)
	    crop_y = int((img_data.shape[1] - self.crop_size)/2)
	    img_data = img_data[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size,:] - self.np_mean[idx%8]
	    clip_frames.append(img_data)

        clip_frames = torch.from_numpy( np.array(clip_frames).astype(np.float32).transpose(3,0,1,2) )

        return clip_frames

    def __len__(self):
        return int((self.random_clip_len-8)/8)

    def get_start_end_frame_id(self):
	return self.start, self.end, float(self.random_clip_len)/self.num_frames
