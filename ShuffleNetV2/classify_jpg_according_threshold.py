# 此脚本根据各张图片的血腥置信度将军警数据集的图片进行划分
import os
import subprocess

label_path = '/home/momo/sun.zheng/Bloody_Recognizing/ShuffleNetV2/label_prob.txt'

with open(label_path, 'r') as f:
    label_dict = []
    for line in f.readlines():
        line = line.strip('\n') # 去掉换行符\n
        b = line.split(' ') # 将每一行以空格为分隔符转换为列表
        label_dict.append(b)

label_dict = dict(label_dict)
#print(label_dict['FC48364D-E741-C439-E4AD-958F9E8D9BBF20181023_L.jpg'])
#print(type(label_dict['FC48364D-E741-C439-E4AD-958F9E8D9BBF20181023_L.jpg']))

#jpg_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/'
jpg_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/norm/'

count = 0
for name in os.listdir(jpg_path):
    count = count + 1
    print(count)
    sor_path = jpg_path + name
    #print(str(sor_path))
    if float(label_dict[name]) >= 0.7:
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/morethan_0.7/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    elif float(label_dict[name]) <= 0.5:
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/lessthan_0.5/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    else:
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/between_0.5_0.7/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)




