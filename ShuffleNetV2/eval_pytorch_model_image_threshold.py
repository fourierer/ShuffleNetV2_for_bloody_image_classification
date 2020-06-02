# encoding: utf-8
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader
#from network import ShuffleNetV1
from network import ShuffleNetV2
import os
import subprocess
import math

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1]的torch.FloadTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 给定均值：(R,G,B) 方差：>（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
])


# 加载模型
print('load ShuffleNetV2 model begin!')
#model = ShuffleNetV1(group=3)
model = ShuffleNetV2()
checkpoint = torch.load('./model_epoch_15.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')


torch.no_grad()

'''
# 以单张图像为单位进行测试
img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/spam_morethan_0.7/A4D261C9-6FBD-481A-997D-C6A83D5EB56D.jpg')
#img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/361C2F0B-7052-F8E4-6BCC-E8B83BB6153020190725_L.jpg')
#img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/FC48364D-E741-C439-E4AD-958F9E8D9BBF20181023_L.jpg')

# 在已经划分为血腥图片的基础上再次进行阈值判定
file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/'
img = data_transform(img)
img = img.unsqueeze(0)
img_ = img.to(device)
outputs = model(img_)
#print(outputs)
#print(type(outputs))
#print(outputs[0][0])
#print(outputs[0][1])

#_, predicted = torch.max(outputs, 1) # 不考虑最大值对应的类别
#print(predicted)
threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
print(threshold)

#with open('/home/momo/sun.zheng/Bloody_Recognizing/ShuffleNetV2/label_prob.txt','a') as file_handle:
    #file_handle.write('FC48364D-E741-C439-E4AD-958F9E8D9BBF20181023_L.jpg' + ' ' + str(threshold))
    #file_handle.write('\n')
'''
#file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/'
file_path = '/home/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/spam/'

count = 0
for name in os.listdir(file_path):
    count = count + 1
    print(count)
    print(name)
    sor_path = file_path + name
    try:
        img = Image.open(sor_path)
        img = data_transform(img)
        img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
        img_ = img.to(device)
        outputs = model(img_)
        threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
    except Exception as e:
        continue

    with open('/home/momo/sun.zheng/Bloody_Recognizing/ShuffleNetV2/label_prob.txt','a') as file_handle:
        file_handle.write(name + ' ' + str(threshold))
        file_handle.write('\n')
    if threshold >= 0.9:
        dst_path = '/home/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/v2_spam_morethan_0.9/'
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    elif threshold > 0.8:
        dst_path = '/home/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/v2_spam_between_0.8_0.9/'
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    elif threshold >0.7:
        dst_path = '/home/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/v2_spam_between_0.7_0.8/'
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    else:
        dst_path = '/home/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/v2_spam_lessthan_0.7/'
        cmd = 'sudo cp ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)


