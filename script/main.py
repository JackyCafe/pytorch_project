import argparse
from pathlib import Path

import torch
from fancy import config as cfg
from torch import linspace, Tensor
from torch.autograd import Variable
from dataset import DataSet
from machine import TrainConfig
import matplotlib.pyplot as plt


def main():
    args = get_arg_parser().parse_args()
    config: TrainConfig = TrainConfig(cfg.YamlConfigLoader(args.train_config))
    x = Variable(linspace(0,100).type(torch.FloatTensor))
    rand = Variable(torch.randn(100))*10
    y = 3*x+rand
    regress = DataSet(x, y, config)
    x_train = regress.x_train.cuda()
    y_train = regress.y_train.cuda()
    a:Tensor = Variable(torch.rand(1).cuda(),requires_grad = True)
    b:Tensor = Variable(torch.rand(1).cuda(),requires_grad = True)
    # a = Variable(torch.rand(len(x_train)).cuda(),requires_grad = True)
    # b = Variable(torch.rand(len(y_train)).cuda(),requires_grad = True)
    print(a.grad)
    learning_rate = 0.0001
    for i in range(1000):
        prediction = a.expand_as(x_train) * x_train + b.expand_as(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        print(f'loss:{loss}')
        loss.backward()
        a.data.add_(-learning_rate * a.grad.data)
        b.data.add_(-learning_rate * b.grad.data)
        a.grad.data.zero_()
        b.grad.data.zero_()

    plt.figure(figsize=(10, 8))
    x_plot, = plt.plot(x_train.cpu(), y_train.cpu(), 'o')
    y_plot, = plt.plot(x_train.cpu(), a.data.cpu() * x_train.cpu() + b.data.cpu(), 'o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def get_arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-t","--train_config", type=Path, default=Path("../configs/train_config.yaml"))
        return parser

if __name__ == '__main__':
    main()