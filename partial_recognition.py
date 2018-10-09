import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn import neighbors
from sklearn import svm

def LoadLabel(filename):
    fp = open(filename,'r')
    labels = [x.strip().split()[2] for x in fp]
    fp.close()
    #return torch.LongTensor(list(map(int,labels)))
    return list(map(int,labels))


bit = 64


# load training data
train_data = np.load(str(bit)+'H_B.npy')
train_label = LoadLabel('./feature_hmdb_TrainSplit1.list')

#load test data
test_data = np.load(str(bit)+'test.npy')
test_label = LoadLabel('./feature_hmdb_TestSplit1.list')


knn = neighbors.KNeighborsClassifier()
#data = np.array(train_data)
#print(train_data.shape)

label = np.array(train_label)
#print(label.shape)
knn.fit(train_data, label)
#print(label)
clf = svm.SVC()
clf.fit(train_data, label)





t = 0.0
for i, data in enumerate(test_data,0):
    #print(data.reshape(1,-1))
    predict = knn.predict(data.reshape(1,-1))
    y_hat = clf.predict(data.reshape(1,-1))
    # print(y_hat)
    # print(test_label[i])
    if y_hat == test_label[i]:
        t+=1

print(t / float(len(test_label)))

train_label = torch.LongTensor(train_label)

test_label = torch.LongTensor(test_label)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(bit, 51)

    def forward(self, x):

        return F.log_softmax(self.l1(x))
        #return self.l5(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

def train(epoch):
    bs = 256
    index = 0
    while (index<len(train_data)):
        start = index
        end = start + bs
        if index+bs >len(train_data):
            end = len(train_data) - 1
        data = train_data[start:end,:]
        target = train_label[start:end]
        #target = torch.LongTensor(target)
        data = torch.from_numpy(data)
        #target = np.argmax(target)
        #print(target)
        #target = target.astype(np.int64)
        #target = torch.LongTensor(target)
        #target = torch.LongTensor(target)
        data = Variable(data)
        target = Variable(target)
        index+=bs
        #print(data)
        #print(target)
        optimizer.zero_grad()
        output = model(data)
        # loss
        loss = F.nll_loss(output, target)
        loss.backward()
        # update
        optimizer.step()
        # if index % 128*10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, index, len(train_data),
        #         100. * index / len(train_data), loss.data[0]))
def test():
    test_loss = 0
    correct = 0.0
    index = 0
    bs = 1
    while (index<len(test_data)):
        start = index
        end = start + bs
        #print(start,end)
        if index+bs > len(test_data):
            end = len(test_data) - 1
        data = test_data[start:end,:]
        target = test_label[start:end]
        #target = torch.LongTensor(target)
        data = torch.from_numpy(data)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        #print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        index+=bs

    test_loss /= len(test_data)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data),
        100.0 *float( correct) / float(len(test_data)+0.0)))

for epoch in range(1,160):
    train(epoch)
    test()

