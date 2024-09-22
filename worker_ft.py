from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler

from build import *
from finetune import *
from run_manager import *


def ft_worker(rank, world_size, path, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.ft_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    ft_set, val_set = build_ft_datasets(local_args.dataset, local_args.args.server)
    ft_sampler = torch.utils.data.distributed.DistributedSampler(ft_set, shuffle=True)
    ft_loader = torch.utils.data.DataLoader(
        ft_set, batch_size=bsz_gpu, num_workers=local_args.run_d.num_workers,
        # pin_memory=True,
        sampler=ft_sampler, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bsz_gpu, num_workers=local_args.run_d.num_workers,
        # pin_memory=True,
        sampler=val_sampler, drop_last=False, persistent_workers=True)

    local_args.val_set_len = len(val_set)

    local_args.model.features_dim = local_args.dataset.num_classes
    model = build_model(local_args.model)
    load_weights(path, model)
    # dim_mlp = model.fc.weight.shape[1]
    # model.fc = nn.Linear(dim_mlp, local_args.dataset.num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    model.cuda()

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank])

    cudnn.benchmark = True

    criterion = build_criterion(local_args.loss.ft).cuda()
    optimizer = build_optimizer(model, local_args.op.ft)

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    start_epoch = 1
    for epoch in range(start_epoch, local_args.run_p.ft_epoch + 1):
        ft_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.ft, optimizer, epoch)
        if local_args.local_rank == 0:
            dm.begin_ft_epoch(optimizer.param_groups[0]['lr'])

        t_loss, v_loss, t_batch_time, v_batch_time, acc1, acc5 = finetune(model, ft_loader, val_loader, optimizer,
                                                                          criterion, local_args, epoch)

        if local_args.local_rank == 0:
            dm.end_ft_epoch(model.module, t_loss, v_loss, t_batch_time, v_batch_time, acc1, acc5)


def ft_worker_amp(rank, world_size, path, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.ft_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    ft_set, val_set = build_ft_datasets(local_args.dataset, local_args.args.server)
    ft_sampler = torch.utils.data.distributed.DistributedSampler(ft_set, shuffle=True)
    ft_loader = torch.utils.data.DataLoader(
        ft_set, batch_size=bsz_gpu, num_workers=local_args.run_d.num_workers,
        # pin_memory=True,
        sampler=ft_sampler, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=bsz_gpu, num_workers=local_args.run_d.num_workers,
        # pin_memory=True,
        sampler=val_sampler, drop_last=False, persistent_workers=True)

    local_args.val_set_len = len(val_set)

    local_args.model.features_dim = local_args.dataset.num_classes
    model = build_model(local_args.model)
    load_weights(path, model)
    # dim_mlp = model.fc.weight.shape[1]
    # model.fc = nn.Linear(dim_mlp, local_args.dataset.num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    model.cuda()

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank])

    cudnn.benchmark = True

    criterion = build_criterion(local_args.loss.ft).cuda()
    optimizer = build_optimizer(model, local_args.op.ft)
    scaler = GradScaler()
    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    start_epoch = 1
    for epoch in range(start_epoch, local_args.run_p.ft_epoch + 1):
        ft_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.ft, optimizer, epoch)
        if local_args.local_rank == 0:
            dm.begin_ft_epoch(optimizer.param_groups[0]['lr'])

        t_loss, v_loss, t_batch_time, v_batch_time, acc1, acc5 = finetune_amp(model, ft_loader, val_loader, optimizer,
                                                                              criterion, local_args, epoch, scaler)

        if local_args.local_rank == 0:
            dm.end_ft_epoch(model.module, t_loss, v_loss, t_batch_time, v_batch_time, acc1, acc5)
