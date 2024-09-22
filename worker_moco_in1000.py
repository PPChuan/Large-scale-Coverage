from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler

from build import *
from pretrain_moco import *
from run_manager_ckp import *


def pretrain_worker_rcrop_amp_in1000(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set = build_rcrop_datasets(local_args.dataset, local_args.args.server)
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )

    local_args.moco.encoder_q = build_model(local_args.model)
    local_args.moco.encoder_k = build_model(local_args.model)
    model = build_model(local_args.moco)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank])

    criterion = build_criterion(local_args.loss.pretrain).cuda()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['moco_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch'] + 1
        data = checkpoint['data']
    else:
        local_args.start_epoch = 1
        data = []

    local_args.data = data

    cudnn.benchmark = True
    scaler = GradScaler()

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        loss, batch_time = pretrain_amp(model, pretrain_loader, optimizer, criterion, local_args, epoch, scaler)

        if local_args.local_rank == 0:
            dm.end_pretrain_epoch(model.module, loss, batch_time)
            dm.save_ckp(model, optimizer)


def pretrain_worker_rcrop_sc_amp_in1000(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set = build_rcrop_datasets(local_args.dataset, local_args.args.server)
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )

    local_args.moco.encoder_q = build_model(local_args.model)
    local_args.moco.encoder_k = build_model(local_args.model)
    model = build_model(local_args.moco)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank])

    criterion = build_criterion(local_args.loss.pretrain).cuda()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    if local_args.args.resume is not None:
        checkpoint = torch.load(os.path.join('./', local_args.args.resume, 'ckp', 'ckp.pth'), map_location='cuda')
        model.load_state_dict(checkpoint['moco_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        local_args.start_epoch = checkpoint['epoch']
        data = checkpoint['data']
    else:
        local_args.start_epoch = 1
        data = []

    local_args.data = data

    cudnn.benchmark = True
    scaler = GradScaler()

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    for epoch in range(local_args.start_epoch, local_args.run_p.pretrain_epoch + 1):

        pretrain_set.epoch = epoch
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        loss, batch_time = pretrain_amp(model, pretrain_loader, optimizer, criterion, local_args, epoch, scaler)

        if local_args.local_rank == 0:
            dm.end_pretrain_epoch(model.module, loss, batch_time)
            dm.save_ckp(model, optimizer)
