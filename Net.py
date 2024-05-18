import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(5, 1000, 3, padding=1)
        self.conv2 = nn.Conv1d(1000, 1000, 3, padding=1)
        self.conv3 = nn.Conv1d(1000, 1000, 3, padding=1)
        self.maxpool = nn.MaxPool1d(25)
        self.avgpool = nn.AvgPool1d(4)
        self.bn20 = nn.BatchNorm1d(1000)
        self.bn40 = nn.BatchNorm1d(1000)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, 100)

    def forward(self, x):  # (N,5,100)
        x = self.bn20(torch.relu(self.conv1(x)))  # (N,1000,100)
        x = self.bn20(torch.relu(self.conv2(x)))  # (N,1000,100)
        x = self.maxpool(x)  # (N,1000,4)
        x = self.dropout10(x)
        x = self.bn40(torch.relu(self.conv3(x)))  # (N,1000,4)
        x = self.avgpool(x)  # (N,1000,1)
        x = self.dropout20(x)
        x = x.view(x.size(0), -1)  # (N,1000)
        x = self.fc(x)  # (N,100)
        return x