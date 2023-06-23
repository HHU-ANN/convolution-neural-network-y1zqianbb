import os

#os.system("sudo pip3 install torch")
#os.system("sudo pip3 install torchvision")
#########

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(  # 输入 32 * 32 * 3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )  # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.dense(x)
        return x


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False,
                                               transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=False,
                                                 transform=torchvision.transforms.ToTensor())
dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False,
                                               transform=torchvision.transforms.ToTensor())
data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)

# 定义全局变量
#modelPath = './model.pkl'
batchSize = 5
nEpochs = 40

model=NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
  # 随机梯度下降
iter = 0
num = 1
    # 训练网络
model.train()
for epoch in range(nEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(data_loader_train, 0):
            iter = iter + 1
            # 取数据
            inputs, labels = data
            inputs, labels = inputs, labels # 将输入和目标在每一步都送入GPU
            # 将梯度置零
            optimizer.zero_grad()
            # 训练
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()   # 反向传播
            optimizer.step()  # 优化
            # 统计数据
            running_loss += loss.item()
            if i % 100 == 99:    # 每 batchsize * 100 张图片，打印一次
                print('epoch: %d\t batch: %d\t loss: %.6f' % (epoch + 1, i + 1, running_loss / (batchSize*100)))
                running_loss = 0.0
                num=num + 1

# 使用测试数据测试网络

correct = 0
total = 0
with torch.no_grad():  # 训练集中不需要反向传播
        for data in data_loader_val:
            images, labels = data
            images, labels = images, labels # 将输入和目标在每一步都送入GPU
            outputs =model(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回每一行中最大值的那个元素，且返回其索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



torch.save(model.state_dict(), 'C:/Users/Lenovo/convolution-neural-network-y1zqianbb/pth/model.pth')

def main():
    model = NeuralNetwork()  # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
