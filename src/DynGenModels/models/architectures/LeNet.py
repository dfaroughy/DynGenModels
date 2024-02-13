import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

#... LeNet architectures

class _ConvNet(nn.Module):
    def __init__(self,  
                 num_classes,
                 num_channels,
                 dim_hidden,
                 filter_size
                 ):
        super(_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels[0], filter_size)
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], filter_size)
        self.fc1 = nn.Linear(num_channels[1] * 4 * 4, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], num_classes)

    def forward(self, x, activation_layer=None):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))
        x = F.relu(self.fc1(x))
        if activation_layer == 'fc1': return x  
        x = F.relu(self.fc2(x))
        if activation_layer == 'fc2': return x 
        x = self.fc3(x)  
        if activation_layer == 'fc3': return x  
        return F.log_softmax(x, dim=1)  

class LeNet3(_ConvNet):
    def __init__(self, num_classes, num_channels, dim_hidden, filter_size=3):
        super(LeNet3, self).__init__(num_classes, num_channels, dim_hidden, filter_size)

class LeNet5(_ConvNet):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__(num_classes, num_channels=(6, 16), dim_hidden=(120, 84), filter_size=5)
