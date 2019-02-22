from scipy import misc
import multiprocessing as mp
import glob
import os

frame_root = '../test_frames'
folder_list = glob.glob(frame_root+'/*')

def fun(folder):
	print folder
	img_list = glob.glob(folder+'/*.jpg')
	for img_name in img_list:
		img = misc.imread(img_name)
		if img.shape[1]>img.shape[0]:
			if img.shape[1] == 112:
				continue
			scale = float(112/float(img.shape[0]))
			img = misc.imresize(img,(int(img.shape[0] * scale + 1), 112))
		else:
			if img.shape[0] == 112:
				continue
			scale = float(112/float(img.shape[1]))
			img = misc.imresize(img,(112, int(img.shape[1] * scale + 1)))
		misc.imsave(img_name, img)
"""
for folder in folder_list:
	print folder
	img_list = glob.glob(folder+'/*.jpg')
	for img_name in img_list:
		img = misc.imread(img_name)
		if img.shape[1]>img.shape[0]:
			if img.shape[1] == 112:
				continue
			scale = float(112/float(img.shape[0]))
			img = misc.imresize(img,(int(img.shape[0] * scale + 1), 112))
		else:
			if img.shape[0] == 112:
				continue
			scale = float(112/float(img.shape[1]))
			img = misc.imresize(img,(112, int(img.shape[1] * scale + 1)))
		misc.imsave(img_name, img)
"""
pool = mp.Pool(processes=15)
pool.map(fun, folder_list)
			
