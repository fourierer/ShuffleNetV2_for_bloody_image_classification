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

device = torch.device('cuda')


# 数据预处理
data_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
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
#model = torch.nn.DataParallel(model)
#checkpoint = torch.load('./2.0x.pth.tar')
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')


torch.no_grad()
# 以单张图像为单位进行测试
# bloody
#img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/bloody/7BBEBDD0-5721-F51E-11EE-FEE8D9D319FC20181026_L.jpg')
# norm
#img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/norm/1354F7D1-2610-2DCC-247D-713729E7093520181107_L.jpg')


norm_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/norm/'
bloody_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test/bloody/'
police_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/police/'
spam_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/val/spam/'

for name in os.listdir(spam_path):
    sor_path = spam_path + name
    print(sor_path)
    img = Image.open(sor_path)
    img = data_transform(img)
    img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
    img_= img.to(device)
    outputs = model(img_)
    #print(outputs)
    _, predicted = torch.max(outputs, 1)
    #print(predicted)
    if predicted==torch.tensor([1], device='cuda:0'):
        print('this picture maybe: norm')
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/norm/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    else:
        print('this picture maybe: bloody')
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/bloody/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
'''
eval_acc = 0
# 批量测试验证集中的图像，使用dataloader，可以更改batch_size调节测试速度
print('test data load begin!')
test_dataset = torchvision.datasets.ImageFolder(root='/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/test', transform=data_transform)
test_data = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)
#print(type(test_data))
print('Test data load done!')
count = 0
for img, label in test_data:
    count = count + 1
    print(count)
    img = img.to(device)
    label = label.to(device)
    out = model(img)

    _, pred = out.max(1)
    #print('pred:' + str(pred))
    #print('label:' + str(label))
    num_correct = (pred == label).sum().item()
    acc = num_correct / img.shape[0]
    print('Test acc in current batch:' + str(acc))
    eval_acc +=acc

print('final acc in Test data:' + str(eval_acc / len(test_data)))
'''


