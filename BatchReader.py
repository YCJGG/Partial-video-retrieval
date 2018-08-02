import numpy as np
import torch.utils.data as data
import torch
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
class DatasetProcessing(data.Dataset):
    def __init__(self, data_path):
        #self.feature_path = data_path
        ffp = open(data_path,'r')
        fp = list(ffp)
        self.fullfeature_path = [x.strip().split(' ')[0] for x in fp]
        self.partialfeature_path = [x.strip().split(' ')[1] for x in fp]
        self.labels = [int(x.strip().split(' ')[2]) for x in fp]
        ffp.close()
        
    def __getitem__(self, index):
        ff = np.load(self.fullfeature_path[index])
        pf = np.load(self.partialfeature_path[index])
        label = torch.LongTensor([self.labels[index]])
        return ff, pf, label, index
    def __len__(self):
        return len(self.fullfeature_path)
#DatasetProcessing('test.list')

class DatasetReader(data.Dataset):
    def __init__(self, data_path):
        liness = open(data_path,'r')
        lines = list(liness)
        self.video_path = [x.strip().split(' ')[0] for x in lines]
        self.labels = [int(x.strip().split(' ')[1]) for x in lines]
        liness.close()

        self.num_frames_per_clip = 16
        self.crop_size = 112
        self.np_mean = np.load('crop_mean.npy').reshape([self.num_frames_per_clip, self.crop_size, self.crop_size, 3])
    

    def __getitem__(self,index):
        filename = self.video_path[index]
        tmp_data = []
        tmp_data_f = []
        s_index = 0
        s_index_f  = 0
        for parent, dirnames, filenames in os.walk(filename):
            if(len(filenames)<self.num_frames_per_clip):
                return [], s_index
            filenames = sorted(filenames)
            s_index = random.randint(0, len(filenames) - self.num_frames_per_clip)
            for i in range(s_index, s_index + self.num_frames_per_clip):
                image_name = str(filename) + '/' + str(filenames[i])
                #print(image_name)
                img = Image.open(image_name)
                img_data = np.array(img)
                tmp_data.append(img_data)
            if self.num_frames_per_clip == 1:
                while (len(ret_arr) < 16):
                    tmp_data.append(img_data)
            while(s_index_f+self.num_frames_per_clip<=len(filenames)):
                arr = []
                for i in range(s_index, s_index + self.num_frames_per_clip):
                
                    image_name = str(filename) + '/' + str(filenames[i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    arr.append(img_data)
                tmp_data_f.append(arr)
                s_index_f += 8
            


        #print(type(filenames))
        #tmp_data, _ = self.get_frames_data(filenames)
        img_datas = [];
        
        #print(len(tmp_data))
        if(len(tmp_data)!=0):
            for j in xrange(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))
                
                if(img.width>img.height):
                    scale = float(self.crop_size)/float(img.height)
                    img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), self.crop_size))).astype(np.float32)
                    
                else:
                    scale = float(self.crop_size)/float(img.width)
                    img = np.array(cv2.resize(np.array(img),(self.crop_size, int(img.height * scale + 1)))).astype(np.float32)
                crop_x = int((img.shape[0] - self.crop_size)/2)
                crop_y = int((img.shape[1] - self.crop_size)/2)
                img = img[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size,:] - self.np_mean[j]
                img_datas.append(img)
        
        data_f = []

        if(len(tmp_data_f)!=0):
            for i in xrange(len(tmp_data_f)):
                imgs_datas_f = []
                for j in xrange(len(tmp_data_f[i])):
                    img = Image.fromarray(tmp_data_f[i][j].astype(np.uint8))
                    
                    if(img.width>img.height):
                        scale = float(self.crop_size)/float(img.height)
                        img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), self.crop_size))).astype(np.float32)
                        
                    else:
                        scale = float(self.crop_size)/float(img.width)
                        img = np.array(cv2.resize(np.array(img),(self.crop_size, int(img.height * scale + 1)))).astype(np.float32)
                    crop_x = int((img.shape[0] - self.crop_size)/2)
                    crop_y = int((img.shape[1] - self.crop_size)/2)
                    img = img[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size,:] - self.np_mean[j]
                    imgs_datas_f.append(img)
                data_f.append(imgs_datas_f)


        
        label = torch.LongTensor([self.labels[index]])

        data = np.array(img_datas).astype(np.float32)

        data_f = np.array(data_f).astype(np.float32)
        #print(data_f.shape)
        data = data.transpose(3, 0, 1, 2)
        data_f = data_f.transpose(0,4,1,2,3)
        
        clip_frames = torch.from_numpy(data)
        all_frame = torch.from_numpy(data_f)

        return clip_frames, label, index, all_frame, filename

    def __len__(self):
        return len(self.video_path)




class DatasetReader2(data.Dataset):
    def __init__(self, data_path):
        liness = open(data_path,'r')
        lines = list(liness)
        self.video_path = [x.strip().split(' ')[0] for x in lines]
        self.labels = [int(x.strip().split(' ')[1]) for x in lines]
        liness.close()

        self.num_frames_per_clip = 16
        self.crop_size = 112
        self.np_mean = np.load('crop_mean.npy').reshape([self.num_frames_per_clip, self.crop_size, self.crop_size, 3])
    

    def __getitem__(self,index):
        filename = self.video_path[index]
        tmp_data = []
        tmp_data_f = []
        s_index = 0
        s_index_f  = 0
        for parent, dirnames, filenames in os.walk(filename):
            if(len(filenames)<self.num_frames_per_clip):
                return [], s_index
            filenames = sorted(filenames)
            s_index = random.randint(0, len(filenames) - self.num_frames_per_clip)
            for i in range(s_index, s_index + self.num_frames_per_clip):
                image_name = str(filename) + '/' + str(filenames[i])
                #print(image_name)
                img = Image.open(image_name)
                img_data = np.array(img)
                tmp_data.append(img_data)
            if self.num_frames_per_clip == 1:
                while (len(ret_arr) < 16):
                    tmp_data.append(img_data)

     #print(type(filenames))
        #tmp_data, _ = self.get_frames_data(filenames)
        img_datas = [];
        
        #print(len(tmp_data))
        if(len(tmp_data)!=0):
            for j in xrange(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))
                
                if(img.width>img.height):
                    scale = float(self.crop_size)/float(img.height)
                    img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), self.crop_size))).astype(np.float32)
                    
                else:
                    scale = float(self.crop_size)/float(img.width)
                    img = np.array(cv2.resize(np.array(img),(self.crop_size, int(img.height * scale + 1)))).astype(np.float32)
                crop_x = int((img.shape[0] - self.crop_size)/2)
                crop_y = int((img.shape[1] - self.crop_size)/2)
                img = img[crop_x:crop_x+self.crop_size, crop_y:crop_y+self.crop_size,:] - self.np_mean[j]
                img_datas.append(img)
            
        label = torch.LongTensor([self.labels[index]])

        data = np.array(img_datas).astype(np.float32)

      
        #print(data_f.shape)
        data = data.transpose(3, 0, 1, 2)
      
        clip_frames = torch.from_numpy(data)
    

        return clip_frames, label, index

    def __len__(self):
        return len(self.video_path)
DatasetReader('./list/test.list')

