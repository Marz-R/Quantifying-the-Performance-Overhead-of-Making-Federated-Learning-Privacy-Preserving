import torch
import torch.nn as nn

class ToyCNN(nn.Module):
    def __init__(self):
        super(ToyCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 channel in, 8 filters
        self.pool = nn.AdaptiveAvgPool2d((4, 4))                # reduce to 4x4
        self.fc   = nn.Linear(8 * 4 * 4, 10)                   # 10 output classes
 
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x