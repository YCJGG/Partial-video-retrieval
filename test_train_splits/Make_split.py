import os
import shutil
split_list = os.listdir('./testTrainMulti_7030_splits')

for split in split_list:
    it = split.split('.')
    if it[0][-1] == '1':
        print('./testTrainMulti_7030_splits/' + split)
        shutil.copyfile('./testTrainMulti_7030_splits/' + split , './split1/'+split )
    elif it[0][-1] == '2':
        print('./testTrainMulti_7030_splits/' + split)
        shutil.copyfile('./testTrainMulti_7030_splits/' + split , './split2/'+split )
    elif it[0][-1] == '3':
        print('./testTrainMulti_7030_splits/' + split)
        shutil.copyfile('./testTrainMulti_7030_splits/' + split , './split3/'+split )