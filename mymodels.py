import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        return x

class Cnn14_4blocks(nn.Module):
    def __init__(self, classes_num=50):
        super(Cnn14_4blocks, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        
    def freeze_backbone(self):
        """冻结所有卷积块和 bn0"""
        for param in self.bn0.parameters():
            param.requires_grad = False
        
        for block in [self.conv_block1, self.conv_block2, 
                      self.conv_block3, self.conv_block4]:
            for param in block.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻（如果需要全参数微调时使用）"""
        for param in self.bn0.parameters():
            param.requires_grad = True
        
        for block in [self.conv_block1, self.conv_block2, 
                      self.conv_block3, self.conv_block4]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        # x 形状: (batch, 1, freq, time) 或 (batch, freq, time)
        # 确保是 4D
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 1, freq, time)
        
        # 原仓库的 forward 逻辑
        x = self.conv0(x)          # (batch, 64, freq, time)
        x = self.bn0(x)
        x = self.conv_block1(x, pool_size=(2,2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2,2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2,2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2,2), pool_type='avg')
        x = torch.mean(x, dim=3)   # 对频率维度平均
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        out = self.fc_audioset(x)  # 去掉 sigmoid，用于 CrossEntropyLoss
        return {'clipwise_output': out}