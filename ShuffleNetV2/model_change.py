import torch
import torch.nn as nn
import copy
#from network import ShuffleNetV2

#model = ShuffleNetV2()
#model = nn.DataParallel(model)
checkpoint = torch.load('./ShuffleNetV2.1.5x.pth.tar')
checkpoint_copy = copy.deepcopy(checkpoint) # 复制checkpoint，并修改

for name, param in checkpoint_copy['state_dict'].items():
    print(name) # 输出各层参数的名称
    if name=='module.classifier.0.weight':
        param.data = torch.ones(2, 1024) # 修改最后一层分类数的权重
        checkpoint_copy['state_dict'][name] = param
    #print('{},size:{}'.format(name, param.data.size()))
torch.save(checkpoint_copy, './pretrained_model_for_bloody.pth')




