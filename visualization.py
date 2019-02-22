import os 
import numpy as np
import shutil 

# os.mkdir('./query')
# os.mkdir('./gallery')
query = np.load('128test.npy')
print(query.shape)
gallery = np.load('128H_B.npy')
print(gallery.shape)

train_file = open('UCF_Train_Retrieval.list')
test_file = open('UCF_Test_Retrieval.list')
train_set = list(train_file)
test_set = list(test_file)


index = 2450

qy = query[index].reshape(128,1)

result = np.dot(gallery,qy).reshape(-1)

#result_  = np.where(result==128)
result_ = np.argsort(result)
result_ = result_[::-1]
re_set = []
re_set_ = []
flag = 0
for re in result_:
    res = train_set[re].split(' ')[0].split('/')
    
    ret = res[2]+'/'+res[3][0:-5]
    
    # if flag == ret:
    #     continue
    # else:
    re_set_.append(ret)
    flag = ret
#print(re_set)

for ret in re_set_:
    if ret not in re_set:
        re_set.append(ret)
    if len(re_set) > 10:
        break


qy_dir = test_set[index]
qy_dir_ = qy_dir.split(' ')[0].split('/')
test_dir = './UCF-101/'+qy_dir_[2]+'/'+qy_dir_[3][0:-5]
print(test_dir)
for i in range(5):
    img = test_dir + '/0000'+ str(i+1) + '.jpg'
    shutil.copy(img,  './temp/query')
    
k = 0
for ind in re_set[0:10] :
    
    #os.mkdir('./gallery/'+str(k))
    #re_dir = test_set[ind]
    #re_dir_ = re_dir.split(' ')[0].split('/')
    #test_dir = './UCF-101/'+re_dir_[2]+'/'+re_dir_[3][0:-5]
    test_dir  = './UCF-101/'+ind
    
    for i in range(5):
        img = test_dir + '/0000'+ str(i+1) + '.jpg'
        print(img)
        shutil.copy(img,  './temp/gallery/'+str(k))
    k+=1