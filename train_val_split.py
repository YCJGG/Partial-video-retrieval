# file1 = open('./features.list','r')
# #file2 = open('./partial_features.list','r')
# file3 = open('./train.list','a')
# file4 = open('./test.list','a')

# file1 = list(file1)
# #file2 = list(file2)

# for i in range(len(file1)):
#     line = file1[i].strip()
#     line= line.split(' ')
#     ff = line[0]
#     pf = line[1]
#     label = line[2]
#     label = label.replace('[','')
#     label = label.replace(']','')
#     #print(label)

#     if i % 13 == 0:
        
#         file4.write(ff+' '+pf+' '+ label+'\n')
#     else:
#         file3.write(ff+' '+pf+' '+ label+'\n')

file1 = open('./list/train.list')
file2 = open('./list/test.list')
file3 = open('./train.list','a')
file4 = open('./test.list','a')

for line in file1:
    line = line.strip().split('//')[1].split(' ')
    ff = './full_features/' + line[0] + '.npy'
    pf = './partial_features/' + line[0] + '.npy'
    label = line[1]
    file3.write(ff+' '+pf+' '+ label+'\n')

for line in file2:
    line = line.strip().split('//')[1].split(' ')
    ff = './full_features/' + line[0] + '.npy'
    pf = './partial_features/' + line[0] + '.npy'
    label = line[1]
    file4.write(ff+' '+pf+' '+ label+'\n')
