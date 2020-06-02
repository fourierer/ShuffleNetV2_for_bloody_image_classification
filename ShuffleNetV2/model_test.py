import torch
import torch.nn as nn
from network import ShuffleNetV2


gpus = [1, 2, 3, 4]

model = ShuffleNetV2()
print(model)
model.load_state_dict(torch.load('./pretrained_model_for_bloody.pth'))
#model = nn.DataParallel(model)
#print(model)

'''
model = ShuffleNetV2()
model = torch.nn.DataParallel(model)
checkpoint = torch.load('./ShuffleNetV2.1.5x.pth.tar')
model.load_state_dict(checkpoint['state_dict']) # 使用训练好的模型进行fine-tune
model.classifier = nn.Sequential(nn.Linear(1024, 2, bias=False))
print(model)
'''
