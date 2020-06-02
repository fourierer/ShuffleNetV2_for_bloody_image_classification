import cv2
import os
import subprocess

bloody_train_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/train/bloody/'
norm_train_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/train/norm/'
bloody_test_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/bloody/'
norm_test_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/norm/'
police_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/police/'

count_current = 0
count_single_channel = 0
file_path = police_path
for img in os.listdir(file_path):
    #print(file_path + img)
    count_current = count_current + 1
    if count_current%100==0:
        print(count_current)
    Img = cv2.imread(file_path + img)
    size = Img.shape
    #print(size)
    #print(size[2])
    if size[2]==1:
        count_single_channel = count_single_channel + 1
        print(count_single_channel)
        cmd = 'sudo rm -f ' + police_path + img
        subprocess.call(cmd, shell=True)


