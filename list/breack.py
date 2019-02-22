import os

file1 = open('train.list','r')
file3 = open('train_b.list','a')

file1 = list(file1)

file2 = file1[10501:]

for line in file2:
    file3.write(line)
