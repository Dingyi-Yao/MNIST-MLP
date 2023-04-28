import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

writer = SummaryWriter('./log')


def load_data_mnist(batch_size, rootpath):
    """
    加载数据集
    """
    # 创建数据集
    training_data =  torchvision.datasets.MNIST(
        root=rootpath, 
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data =  torchvision.datasets.MNIST(
        root=rootpath, 
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # 创建 dataloader
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits



def train(dataloader, model, loss_fn, optimizer):
    """
    训练模型
    """

    size = len(dataloader.dataset)
    for batch, (img, label) in enumerate(dataloader):

        img, label = img.to(device), label.to(device)
        pred = model(img)
        loss = loss_fn(pred, label)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # 每100个batch打印一次loss
        # if batch % 100 == 0:
        #     print("训练损失" ,loss.item())

    return loss.item()
#定义测试集
def test(dataloader, model):
    """
    测试模型
    """
    size = len(dataloader.dataset)
    #关闭dropout和batchnormalization
    model.eval()
    test_loss, correct_num = 0, 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            #统计损失
            test_loss_single = loss_fn(pred, label).item()
            test_loss += test_loss_single
            #统计数量
            correct_num += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss_avg = test_loss/size
    acc = correct_num / size
    print("正确率",acc)
    print("测试平均损失",test_loss_avg)
    return acc, test_loss_avg

if __name__ == '__main__':
    #参数设置
    
    batch_size = 64
    rootpath='F:\\dataset\\data\\MNIST\\raw'
    device = "cuda" 
    print("Using {} device".format(device))
    epochs = 100

    #模型设置
    model = Model().to(device)
    # print(model)
    images = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, images)

    #优化器设置
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #加载数据
    train_dataloader, test_dataloader=load_data_mnist(batch_size, rootpath)

    #训练与验证模型
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        writer.add_scalar('train_loss', train_loss, t)
        acc, test_loss_avg = test(test_dataloader, model)
        #记录数据
        writer.add_scalar('test_loss_avg', test_loss_avg, t)
        writer.add_scalar('acc', acc, t)
    print("Done!")

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

