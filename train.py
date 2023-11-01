import os
from functools import partial

import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.callback import LossHistory
import torch.optim as optim
from utils.utils import get_num_classes, seed_everything, show_config, worker_init_fn
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
import torch
import torch.distributed as dist
from nets.facenet import Facenet
from nets.facenet_training import triplet_loss, get_lr_scheduler, set_optimizer_lr

from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    seed = 11
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置           distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置           distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    distributed = False    # 是否使用单机多卡分布式运行，终端指令仅支持Ubuntu
    sync_bn = False    # sync_bn 是否使用sync_bn，DDP模式多卡可用
    fp16 = False    # 是否使用混合精度训练。可减少约一半的显存、需要pytorch1.7.1以上

    annotation_path = "cls_train.txt"    # 指向根目录下的cls_train.txt，读取人脸路径与标签
    input_shape = [160, 160, 3]    # 输入图像大小，常用设置如[112, 112, 3]
    #   主干特征提取网络的选择
    backbone = "mobilenet"
    #   当model_path = ''的时候不加载整个模型的权值。
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    model_path = "model_data/facenet_mobilenet.pth"
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，此时从0开始训练。
    pretrained = False

    # 训练参数
    batch_size = 96    # 需要为3的倍数
    Init_Epoch = 0
    Epoch = 100

    # 其它训练参数：学习率、优化器、学习率下降有关
    Init_lr = 1e-3    # 模型的最大学习率
    Min_lr = Init_lr * 0.01    # 模型的最小学习率
    optimizer_type = "adam" # 当使用Adam优化器时建议设置  Init_lr=1e-3； 当使用SGD优化器时建议设置   Init_lr=1e-2
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = "cos"    # 使用到的学习率下降方式，可选的有step、cos
    save_period = 1    # 多少个epoch保存一次权值
    save_dir = 'logs'    # 权值与日志文件保存的文件夹
    num_workers = 4
    lfw_eval_flag = True    # 是否开启LFW评估
    lfw_dir_path = "lfw"  # LFW评估数据集的文件路径和对应的txt文件
    lfw_pairs_path = "model_data/lfw_pair.txt"


#-------------------------------------------------------------------------------------------------------------------
    # 1.设置种子；2.显卡数量设置；3.分布式
    # 载入模型  加载预训练权重
#-------------------------------------------------------------------------------------------------------------------
    seed_everything(seed)

    ngpus_per_node = torch.cuda.device_count()  # 返回GPU的数量

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        local_rank      = 0
        rank            = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # 载入模型, 加载预训练权重
    num_classes = get_num_classes(annotation_path)
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        #   根据预训练权重的Key和模型的Key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict) # 载入权重
        if local_rank == 0: # 显示没有匹配上的Key
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    loss = triplet_loss()
    # 实例化一个 LOSS记录的LossHistory类
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None


    # 将实例化后的model切换为train模式，并赋值为model_train
    model_train = model.train()
    # 如果是多卡就进行相应设置
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    # 设置DP或者DDP
    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)    # 多卡平行运行
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #   LFW估计 的 dataloader
    LFW_loader = DataLoader(dataset=LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape),
                            batch_size=32, shuffle=False) if lfw_eval_flag else None




# 划分训练和验证用的数据
    #   0.01用于验证，0.99用于训练
    val_split = 0.01
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )


    if True:
        if batch_size % 3 != 0:
            raise ValueError("Batch_size must be the multiple of 3.")
        #  根据当前batch_size，自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #   根据optimizer_type选择优化器
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        #   判断每一个世代的长度(每一个epoch下有多少step)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        #   实例化数据集。
        train_dataset = FacenetDataset(input_shape, lines[:num_train], num_classes, random=True)
        val_dataset = FacenetDataset(input_shape, lines[num_train:], num_classes, random=False)
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        #   实例化数据集加载器。
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size // 3, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size // 3, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen,
                          gen_val, Epoch, Cuda, LFW_loader, batch_size // 3, lfw_eval_flag, fp16, scaler, save_period,
                          save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()