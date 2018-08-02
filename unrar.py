import os

rar_list = os.listdir('./')
for rar in rar_list:
    if rar[-3:] == 'rar':
        os.system('unrar x '+rar)