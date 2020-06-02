import os
import subprocess

# 血腥数据集划分
print('bloody begin')
bloody_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/all_data/bloody/'
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/'
count = 0

for img in os.listdir(bloody_path):
    count = count + 1
    sor_path = bloody_path + img
    if count%5==0:
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path + 'test/bloody/'
    else:
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path + 'train/bloody/'
    subprocess.call(cmd, shell=True)
print('bloody done')

print('norm begin')
# 非血腥数据集划分
norm_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/all_data/norm/'
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/'
count = 0

for img in os.listdir(norm_path):
    count = count + 1
    sor_path = norm_path + img
    if count%5==0:
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path + 'test/norm/'
    else:
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path + 'train/norm/'
    subprocess.call(cmd, shell=True)
print('norm done')

