import os

lr = 1e-3*2
k = 0
while(k<9):
    lr = lr/2
    os.system('python fine_tune.py ' + '--lrC='+str(lr))
    k += 1
       
