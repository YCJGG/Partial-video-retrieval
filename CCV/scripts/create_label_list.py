import glob, os

phase = 'train'
def for_full_feature(only_one_hot = False):
	feature_dir = '../full_{}_features/'.format(phase)
	lines_id = file('../{}VidID.txt'.format(phase)).readlines()
	lines_label = file('../{}Label.txt'.format(phase)).readlines()
	if only_one_hot == False:
		f = file('../full_{}_id_label.txt'.format(phase),'w')
	else:
		f = file('../filtered_full_{}_id_label.txt'.format(phase),'w')
	s = ''
	for id, label in zip(lines_id, lines_label):
		if only_one_hot == True and label.count('1') > 1:
			continue
		if os.path.isfile(feature_dir + id.strip() + '.npy') == True:
			s += feature_dir + id.strip() + '.npy '  +  label
	f.write(s)

def for_partial_feature(only_one_hot = False):
	feature_dir = '../partial_{}_features/'.format(phase)
	lines_id = file('../{}VidID.txt'.format(phase)).readlines()
	lines_label = file('../{}Label.txt'.format(phase)).readlines()
	if only_one_hot == False:
		f = file('../partial_{}_id_label.txt'.format(phase),'w')
	else:
		f = file('../filtered_partial_{}_id_label.txt'.format(phase),'w')
	s = ''
	for id, label in zip(lines_id, lines_label):
		if only_one_hot == True and label.count('1') > 1:
			continue
		for i in range(10):
			if os.path.isfile(feature_dir + id.strip() + '_' + str(i) + '.npy') == True:
				s += feature_dir + id.strip() + '_{}.npy '.format(i)  +  label
	f.write(s)
	f.close()

for_partial_feature(only_one_hot = True)
