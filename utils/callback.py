import datetime
import os
import torch

import matplotlib
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import scipy.signal

class LossHistory():
    def __init__(self, log_dir, model, input_shape):

        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')    # 获取时间，并新建一个logs的文件夹
        self.log_dir = os.path.join(log_dir, "loss_" + str(time_str))
        os.makedirs(self.log_dir)

        self.acc = []
        self.losses = []
        self.val_loss = []

        self.writer = SummaryWriter(self.log_dir)   # 实例化一个SummaryWriter类
        dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
        self.writer.add_graph(model, dummy_input)   # 可视化模型计算图

    def append_loss(self, epoch, acc, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.acc.append(acc)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
            f.write(str(acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        # scipy.signal.savgol_filter是进行滤波，从而平滑化。
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")    # 图示标签位置
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

        plt.figure()
        plt.plot(iters, self.acc, 'red', linewidth=2, label='lfw acc')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.acc, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth lfw acc')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Lfw Acc')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
        plt.cla()
        plt.close("all")
