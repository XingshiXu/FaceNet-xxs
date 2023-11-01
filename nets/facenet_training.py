import math
import numpy as np
import torch
from functools import partial    # 它对普通函数进行封装， 主要功能是把一个函数的部分参数给固定住，返回一个新的函数。通俗点说， 就是冻结原函数的某些参数。


def triplet_loss(alpha=0.2):
    def _triplet_loss(y_pred, Batch_size):
        anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[int(2 * Batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1)) # a 与 p 的距离
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1)) # a 与 n 的距离
                                            # pow 是求指数的操作；pow(input, exponent)

        # 查找困难三元组
        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten() # neg_dist比pos_dst大时False;;;;;反之为True
        hard_triplets = np.where(keep_all == 1)    # 保留neg_dist比pos_dst小的，设置为True

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
        return loss

    return _triplet_loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:    #step
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    # 动态修改学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


## 研究triplet_loss代码时写的代码，没有用
# neg_dist = torch.tensor([1,2,3,4,5])
# rrr_dist = 3
# keep_all = (neg_dist-rrr_dist<0).cpu().cpu().numpy().flatten()
# hard_triplets = np.where(keep_all == 1)
# neg_dist1 = neg_dist[hard_triplets]
# print("6")
