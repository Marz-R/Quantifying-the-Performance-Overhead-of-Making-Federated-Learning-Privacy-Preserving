import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes, in_channels, fc_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128*fc_size, 128)  # After 3 pooling layers, image size reduces to 4x4 (cifar) or 3x3(mnist)
        self.fc2 = nn.Linear(128, num_classes) # output
    
    # Progresses data across layers    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(self.pool(x))
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(self.pool(x))
        x = self.dropout(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.bn3(x)
        x = F.relu(self.pool(x))
        x = self.dropout(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x