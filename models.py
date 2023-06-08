import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(.05)
        self.conv2 = nn.Conv2d(128, 8, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm2d(8)
        #self.dropout1 = nn.Dropout(.05)
        self.conv3 = nn.Conv2d(8, 64, kernel_size= 3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1)
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 10, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2,2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc1 = nn.Linear(128, 10)

        

    def forward(self, x):
        x = self.conv1(x) # ==> [512,128,28,28] >> RF : 3
        x = F.relu(x)
        x = self.batchnorm1(x)
        #x = self.dropout1(x)
        x = self.pool1(x) # ==>[512,128,14,14] >> RF: 4
        x = self.conv2(x) # ==>[512,8,14,14] >> RF: 4
        x = F.relu(x)
        x = self.batchnorm2(x)
        #x = self.dropout1(x)
        x = self.conv3(x) # ==>[512,64,12,12] >> RF: 8
        x = F.relu(x)
        x = self.batchnorm3(x)
        #x = self.dropout1(x)
        x = self.pool1(x) # ==>[512,64,6,6] >> RF: 9
        x = self.conv4(x) # ==> [512,32,6,6] >> RF: 9
        x = F.relu(x)
        x = self.batchnorm4(x)
        #x = self.dropout1(x)
        x = self.conv5(x) # ==> [512,16,4,4] >> RF: 17
        x = F.relu(x)
        x = self.batchnorm5(x)
        #x = self.dropout1(x)
        x = self.conv6(x)# ==> [512, 10, 2, 2] >> RF: 25
        x = F.relu(x)
        x = self.batchnorm6(x)
        #x = self.pool1(x)
        x = self.avg_pool(x) # ==> [512, 10, 1, 1]
        #pdb.set_trace()
        x = x.view(-1, 10)
        #x = self.fc1(x)
        return F.log_softmax(x, dim=1)
