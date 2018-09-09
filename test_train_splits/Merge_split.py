import os
trainFile = open('TrainSplit3.txt','a')
testFile = open('TestSplit3.txt','a')
file_list = os.listdir('./split3')
k = 0
#print(file_list)
path = '/home/zhangjingyi/Rescode/hash/Partial-video-retrieval/HMDB51/'
for filename in file_list:
    file = open('./split3/'+filename,'r')
    dir = filename.split('test')
    dir = dir[0][:-1]
    #print(dir)
    for line in file:
        cont = line.strip().split(' ')
        if cont[1] == '1':
            trainFile.write(path + dir +'/'+ cont[0][:-4]+' '+str(k)+'\n')
        elif cont[1] == '2':
            testFile.write(path + dir +'/'+cont[0][:-4]+' '+str(k)+'\n')
    k+=1
    file.close()
