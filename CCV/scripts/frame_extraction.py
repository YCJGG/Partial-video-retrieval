import os,glob
import pdb
import cv2
import multiprocessing as mp
from scipy import misc

video_dir = '../test'
frame_root = '../test_frames'
video_list = glob.glob(video_dir+'/*.mp4')
info_file = file('test_frames_info.txt','w')

def fun(enu_video_list):
	idx, vid = enu_video_list
	vid_name = vid.split('/')[-1].split('.')[0]
	print 'Processing {}, {}/{}'.format(vid_name, idx, len(video_list))

	frame_dir = frame_root + '/' + vid_name
	if os.path.isdir(frame_dir) == False:
		os.mkdir(frame_dir)

	video = cv2.VideoCapture(vid)
	fps = video.get(cv2.CAP_PROP_FPS)

	frame_count = 0
	while True:
		ret, frame = video.read()
		if ret is False:
			break

		if frame.shape[1]>frame.shape[0]:
			if frame.shape[1] == 112:
				continue
			scale = float(112/float(frame.shape[0]))
			frame = misc.imresize(frame,(int(frame.shape[0] * scale + 1), 112))
		else:
			if frame.shape[0] == 112:
				continue
			scale = float(112/float(frame.shape[1]))
			frame = misc.imresize(frame,(112, int(frame.shape[1] * scale + 1)))

		cv2.imwrite(frame_dir + '/{:05}'.format(frame_count) + '.jpg', frame)
		frame_count += 1

	return vid_name + '\t' + str(fps) + '\t' + str(frame_count) + '\n'

pool = mp.Pool(processes=38)
info_s = pool.map(fun, enumerate(video_list))

s = ''
for i in info_s:
	s+=i

info_file.write(s)
info_file.close()
