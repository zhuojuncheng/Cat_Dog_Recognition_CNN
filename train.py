import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils import data
from data import dataset
from MyNet import MyNet


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 1
    batch_size = 200

    cat_dog_data = dataset('img')
    train_data = data.DataLoader(dataset=cat_dog_data, batch_size=batch_size, shuffle=True)
    net = MyNet().to(device)

    optimizer = torch.optim.Adam(net.parameters())
    loss_fun = nn.MSELoss()
    losses = []

    for i in range(epochs):
        print('epochs:{}'.format(i))
        for j, (input, target) in enumerate(train_data):
            input = input.to(device)
            output = net(input)
            target = target.long()
            target = torch.zeros(target.size()[0], 2).scatter_(1, target.view(-1, 1), 1)
            loss = loss_fun(target, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 10 == 0:
                losses.append(loss.float())
                print('[epochs-{0}-{1}/{2}] loss:{3}'.format(i, j, len(train_data), loss.float()))
    torch.save(net, 'models/net.pth')


