import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib2 import Path
from torch.autograd import Variable

from dataset import DataSet
from machine import TrainConfig
from fancy import config as cfg


def main():
    data_path = '../data/hour.csv'
    rides = pd.read_csv(data_path)
    counts = rides['cnt'][:50]
    args = get_arg_parser().parse_args()
    config: TrainConfig = TrainConfig(cfg.YamlConfigLoader(args.train_config))

    var_x = torch.FloatTensor(np.arange(len(counts), dtype=float))
    var_y = torch.FloatTensor(np.array(counts,dtype=float))
    regress = DataSet(var_x, var_y, config)
    print(regress.y_train.data)
    # x = Variable(torch.FloatTensor().cuda())


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_config", type=Path, default=Path("../../configs/train_config.yaml"))
    return parser


if __name__ == '__main__':
    main()
