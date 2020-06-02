# ShuffleNetV2_for_bloody_image_classification
Using ShuffleNetV2 for bloody image classification, train from scratch.

使用移动端小网络ShuffleNetV2来分类bloody图片

1.将散乱的图像整理为规则的训练集和测试集

原数据集中的图像分布在各个文件夹当中，需要先整理到统一的文件夹中，再对统一的文件夹中的图像进行测试集和训练集的划分。

代码：data_preparation.py抽取bloody/data/中两个类别中的图片全部抽取出来；

```python
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
```

代码：data_divide.py将抽取出来的所有图片进行测试集和训练集的划分；

```python
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
```





2.将数据中与训练测试无关的文件剔除

数据集划分之后准备开始训练，训练过程中由于有一些脏乱文件使得训练报错，需要剔除这些文件。代码image_check.py，对训练集中和测试集中所有的图像使用cv进行读取测试，看是否是正常的图片，输出不正常的图片路径进行删除；

```python
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


```





3.对数据进行训练（代码ShuffleNetV2/multiprocessing_distributed.py）

ShuffleNetV2训练结果如下：

epoch 0：84.701%

epoch 1：85.826%

epoch 2：88.058%

epoch 3：88.883%

epoch 4：88.963%

epoch 5：89.974%

epoch 6：90.815%

epoch7：90.636%

......

epoch24：93.326%

epoch25：93.387%

epoch26：93.573%

epoch27：93.550%

epoch28：93.614%

epoch29：93.583%

训练完毕，保存分类准确率为93.614%的模型。



GhostNet训练结果如下：

epoch 0：82.587%

epoch 1：83.873%

epoch 2：84.931%

...

epoch 4：88.171%

epoch 5：86.602%

epoch 6：88.487%

epoch 7：89.000%

......

epoch 20：93.070%

epoch 21：92.998%

epoch22：93.412%

epoch 23：93.382%

......

epoch 27：93.896%

epoch 28：93.911%

epoch 29：94.138%

训练完毕，保存分类准确率为94.138%的模型。



4.利用保存的模型来测试其它数据集中的数据，看是否会发生模型误伤的情况

（1）测试police中的图片，在val文件夹中新建police文件夹，将图片全部放到police中，然后进行测试，测试划分结果存于result/police中；

（2）测试police数据集，进行是否为血腥图片的分类；

测试代码:eval_pytorch_model_image.py进行单张图片的识别；

```python
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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

for name in os.listdir(police_path):
    sor_path = police_path + name
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
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/norm/'
        cmd = 'sudo mv ' + sor_path + ' ' + dst_path
        subprocess.call(cmd, shell=True)
    else:
        print('this picture maybe: bloody')
        dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/'
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


```

在测试时，会发生两种情况的报错：

（1）由于测试的图像是单通道的，输入网络时和网络的输入不一致报错

```python
RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
```

（2）由于测试的图像发生破损，输入网络时无法使用transform来转换报错

```python
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
```

遇到这种图片直接删除，再次运行脚本测试剩下的图片即可。



5.误伤情况优化

4中代码有两个问题需要改变：

（1）4中使用训好的模型测试了police中所有图像，测试的方式为输出的tensor中哪个分量大，就判定为哪一类，如输出的tensor为$(-1,2)$，则判定为第2类，这种判定方法实际上阈值设置为了0.5。要想得到更好的分类判定，需要先将输出映射为概率，如$(\frac{e^{-1}}{e^{-1}+e^{2}},\frac{e^{2}}{e^{-1}+e^{2}})$，此时的输出概率理解为判定为某一类的置信度，需要设置一下阈值来判定是否为血腥图片；

（2）4中测试的方法里面包含了随机裁剪，这是不合适的，训练的时候随机裁剪是为了数据增广（包括在训完一个epoch之后的测试也没有随机裁剪，大部分时候而是先resize到256，再裁减到224）。这里为了测试一副图像的真实分类，不进行任何裁剪操作，只需要resize到224即可。

基于以上两点优化测试代码直接输出军警数据集中每张图的血腥置信度保存为label_prob.txt文件（eval_pytorch_model_image_threshold.py）：

```python
# 添加阈值来判定是否为血腥图像
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
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 固定batchnorm，dropout等，一定要有
model = model.to(device)
print('load ShuffleNetV2 model done!')


torch.no_grad()

'''
# 以单张图像为单位进行测试
#img = Image.open('/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/bloody/C9BCE7FF-B624-688C-BC50-828191049DD420190629_L.jpg')
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
file_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/police/norm/'
count = 0
for name in os.listdir(file_path):
    count = count +1
    print(count)
    sor_path = file_path + name
    img = Image.open(sor_path)
    img = data_transform(img)
    img = img.unsqueeze(0) # 这里直接输入img不可，因为尺寸不一致，img为[3,224,224]的Tensor，而模型需要[1,3,224,224]的Tensor
    img_= img.to(device)
    outputs = model(img_)
    threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
    with open('/home/momo/sun.zheng/Bloody_Recognizing/ShuffleNetV2/label_prob.txt','a') as file_handle:
        file_handle.write(name + ' ' + str(threshold))
        file_handle.write('\n')
```



利用生成的图片血腥置信度标签将军警数据集中的图片划分为3类，置信度<=0.5，0.5<置信度<0.7，置信度>=0.7，脚本classify_jpg_according_threshold.py



6.测试/mnt/data1/dataset/spam/下四个文件夹中所有的图片

由于这个文件夹中的图片过多（260多万），不可能像上面那样出现一张有问题的图片手动删除后再继续运行脚本测试，下面给出一个具有鲁棒性（使用try except语句）的测试代码test.py，直接测试一个文件夹下的所有图片，不需要中途手动调试代码。

```python
# encoding: utf-8
import torch
import torchvision
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from torch.utils.data import DataLoader
from network import ShuffleNetV2
import subprocess
import os
import math


device = torch.device('cuda')

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 加载模型
print('load ShuffleNetV2 model begin!')
model = ShuffleNetV2()
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.to(device)
print('load ShuffleNetV2 model done!')

count = 0
# 测试某文件夹下的所有图片
print('spam begin')
spam_file_path = '/mnt/data1/dataset/spam/'
for dirpath, dirname, filenames in os.walk(spam_file_path):
    if len(filenames) == 0: # 当前文件夹没有文件，则进入下一个有文件的文件夹
        continue
    for name in filenames:
        if name.endswith('.jpg'): # 当前文件为图片
            sor_path = dirpath + '/' + name
            count = count + 1
            print(count)
            print(name)
            try:
                img = Image.open(sor_path)
                img = data_transform(img)
                img = img.unsqueeze(0)
                img_ = img.to(device)
                outputs = model(img_)
                threshold = math.exp(outputs[0][0])/(math.exp(outputs[0][0]) + math.exp(outputs[0][1]))
            except Exception as e:
                continue
            if threshold >= 0.7:
                dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/spam_morethan_0.7/'
                cmd = 'sudo cp ' + sor_path + ' ' + dst_path
                subprocess.call(cmd, shell=True)
            elif threshold <= 0.5:
                dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/spam_lessthan_0.5/'
                cmd = 'sudo cp ' + sor_path + ' ' + dst_path
                subprocess.call(cmd, shell=True)
            else:
                dst_path = '/mnt/data3/cheng.zhaoxi/DTS-data/data/bloody/data/result/spam/spam_between_0.5_0.7/'
                cmd = 'sudo cp ' + sor_path + ' ' + dst_path
                subprocess.call(cmd, shell=True)

print('spam done')
```



7.模型优化

7.1误分的图像可以划分为如下几类：

（1）红色纹身或者皮肤病类：该类与人体联系最为紧密，并且从人眼的角度来看，红色纹身等也非常容易被视为血腥图片，所以这一类误分可以视为是血腥的图片，不参与后续的模型迭代；

（2）装扮类，如穿着红色衣服，涂红色指甲油、红色口红，手上有红色配饰等：这一类是模型亟须改善的一类，原因是这一类从人的角度很容易判断为非血腥，但是计算机又容易误判，所以增加这一类的样本来迭代模型可以使模型理解特定场景下的红色为非血腥；

（3）人和红色物体相对独立，叠加类：这一类图片在血腥置信度阈值为0.5-0.7之间也有很多，所以这一类后续在模型改善之后可以通过一定程度上提高置信度阈值来滤掉；

（4）红色物体类（没有人物出现，只有文字，图案或者其他）：红色物体类和第（2）类一样都是需要模型改善的，容易被人分辨，但是计算机容易误分，所以应当提高这类样本的比例进行模型矫正；（5）裸露色情类：该类应该是由于spam文件夹中的图片特性导致，在阈值小于0.5和0.5到0.7之间的类别中也有很多裸露色情的，所以这一类并不是一定误分的，后续可以暂时不考虑。

综上的分析过程，第二次模型迭代训练时，应该重点考虑（2）装扮类和（4）红色物体类，二者均属于人眼容易区分，但计算机容易误分的情形，需要加大训练样本来改善模型。



在网上爬取相关的图片加入到训练集当中，有以下几类：

（1）red_clothes：包括红色衣服以及穿红色衣服的人，共867张图片；

（2）red_eye：包括红色眼妆，共465张；

（3）red_jewelry：包括单纯的红色配饰和穿戴红色配饰的人，共904张图片；

（4）red_nail：包括单纯的红色指甲油和涂红色指甲油的指甲，共656张图片；

（5）red_object：包括红色物体和红色水果，共1214张图片；

（6）lipstick：包括红色口红和涂红色口红的嘴唇，共1021张图片；

（7）red_logo：包括红色图案和红色文字，共789张图片；

（8）red_hair：包括染红色头发的，共1055张图片；

重新微调模型，初始学习率为0.05，微调20个epochs，训练结果如下：

......

epoch 10：93.711%

epoch 11：94.021%

epoch 12：93.565%

epoch 13：93.774%

epoch 14：94.087%

epoch 15：94.131%

epoch 16：94.228%

epoch 17：94.116%

epoch 18：94.329%

epoch 19：94.410%



7.2使用第二版的模型测试spam数据集中的所有图片，发现误分图片的数量比第一版的模型少了很多。但是在计算置信度大于0.8和大于0.9区间的准确率时，发现准确率还是比较低，轻量级网络ShuffleNetV2是否适用于血腥图片的识别有待进一步研究。