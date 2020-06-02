import subprocess
import os

'''
# 血腥图片的整理
print('bloody begin')
bloody_file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/bloody-20191101/' # 血腥数据集所在目录
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/all_data/bloody/'

for dirpath, dirname, filenames in os.walk(bloody_file_path):
    # print(dirpath, dirname, filenames) # 遍历当前文件夹下所有文件夹以及文件，分别为字符串，字典，字典
    if len(filenames) == 0: # 当前文件夹没有图片，则进入下一个有图片的文件夹
        continue
    for img in filenames:
        #print(dirpath + '/' + img) # 输出所有图片的路径
        sor_path = dirpath + '/' + img
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
print('bloody done')



# 非血腥图片的整理
print('norm begin')
norm_file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/norm-20191101/' # 血腥数据集所在目录
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/all_data/norm/'

for dirpath, dirname, filenames in os.walk(norm_file_path):
    # print(dirpath, dirname, filenames) # 遍历当前文件夹下所有文件夹以及文件，分别为字符串，字典，字典
    if len(filenames) == 0: # 当前文件夹没有图片，则进入下一个有图片的文件夹
        continue
    for img in filenames:
        #print(dirpath + '/' + img) # 输出所有图片的路径
        sor_path = dirpath + '/' + img
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
print('norm done')
'''
'''
# police图片的整理
print('police begin')
police_file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/police/data/' # police图片所在目录
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/police/' # 将所有police图片重新复制一份到该目录下

for dirpath, dirname, filenames in os.walk(police_file_path):
    # print(dirpath, dirname, filenames) # 遍历当前文件夹下所有文件夹以及文件，分别为字符串，字典，字典
    if len(filenames) == 0: # 当前文件夹没有文件，则进入下一个有文件的文件夹
        continue
    for img in filenames:
        if img.endswith('.jpg'): # 当前文件为图片
            #print(dirpath + '/' + img) # 输出所有图片的路径
            sor_path = dirpath + '/' + img
            cmd = 'sudo cp ' + sor_path + ' ' + dst_path
            subprocess.call(cmd, shell=True)
print('police done')
'''
'''
# fuwu_patch图片的整理
print('fuwu_patch begin')
fuwu_patch_file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/fuwu_patch/data/' # fuwu_patch图片所在目录
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/fuwu_patch/' # 将所有fuwu_patch图片重新复制一份到该目录下

for dirpath, dirname, filenames in os.walk(fuwu_patch_file_path):
    # print(dirpath, dirname, filenames) # 遍历当前文件夹下所有文件夹以及文件，分别为字符串，字典，字典
    if len(filenames) == 0: # 当前文件夹没有文件，则进入下一个有文件的文件夹
        continue
    for img in filenames:
        if img.endswith('.jpg'): # 当前文件为图片
            #print(dirpath + '/' + img) # 输出所有图片的路径
            sor_path = dirpath + '/' + img
            cmd = 'sudo cp ' + sor_path + ' ' + dst_path
            subprocess.call(cmd, shell=True)
print('fuwu_patch done')
'''

# spam图片的整理
print('spam begin')
spam_file_path = '/mnt/data1/dataset/spam/'
dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/spam/' # 将所有spam中的图片重新复制一份到该目录下
for dirpath, dirname, filenames in os.walk(spam_file_path):
    if len(filenames) == 0: # 当前文件夹没有文件，则进入下一个有文件的文件夹
        continue
    for img in filenames:
        if img.endswith('.jpg'): # 当前文件为图片
            sor_path = dirpath + '/' + img
            cmd = 'sudo cp ' + sor_path + ' ' + dst_path
            subprocess.call(cmd, shell=True)
print('fuwu_patch done')















