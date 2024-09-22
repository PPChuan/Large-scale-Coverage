from torch.backends import cudnn

from build import *
from pretrain_byol import pretrain
from run_manager import *


def pretrain_worker_rcrop(rank, world_size, cfg):
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

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cpu()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    # cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    start_epoch = 1
    for epoch in range(start_epoch, local_args.run_p.pretrain_epoch + 1):
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])
        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)
        if local_args.local_rank == 0:
            dm.end_pretrain_epoch(model.module, loss, batch_time)


def pretrain_worker_rcrop_sc(rank, world_size, cfg):
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

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cpu()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    # cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    start_epoch = 1
    for epoch in range(start_epoch, local_args.run_p.pretrain_epoch + 1):
        pretrain_set.epoch = epoch
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])
        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)
        if local_args.local_rank == 0:
            dm.end_pretrain_epoch(model.module, loss, batch_time)


def pretrain_worker_ccrop(rank, world_size, cfg):
    print('==> Start rank:', rank)
    local_args = copy.copy(cfg)
    local_args.local_rank = rank % world_size

    torch.cuda.set_device(local_args.local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{local_args.port}',
                            world_size=world_size, rank=rank)

    bsz_gpu = int(local_args.run_p.pretrain_batch_size / world_size)
    print('batch_size per gpu:', bsz_gpu)
    local_args.bsz = bsz_gpu

    pretrain_set, eval_set = build_ccrop_datasets(local_args.dataset, local_args.args.server)
    len_ds = len(pretrain_set)

    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set, shuffle=True)
    pretrain_loader = torch.utils.data.DataLoader(
        pretrain_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=pretrain_sampler,
        drop_last=True
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_set, shuffle=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=bsz_gpu,
        num_workers=local_args.run_d.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
        drop_last=False
    )

    local_args.byol.encoder_q = build_model(local_args.model)
    local_args.byol.encoder_k = build_model(local_args.model)
    model = build_model(local_args.byol)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_args.local_rank],
                                                      find_unused_parameters=True)

    criterion = build_criterion(local_args.loss.pretrain).cuda()
    optimizer = build_optimizer(model, local_args.op.pretrain)

    cudnn.benchmark = True

    if local_args.local_rank == 0:
        dm = DataManager(local_args)
    start_epoch = 1
    for epoch in range(start_epoch, local_args.run_p.pretrain_epoch + 1):
        pretrain_sampler.set_epoch(epoch)
        adjust_learning_rate(local_args.lr_cfg.pretrain, optimizer, epoch)

        if local_args.local_rank == 0:
            dm.begin_pretrain_epoch(optimizer.param_groups[0]['lr'])

        pretrain_set.use_box = epoch >= local_args.box.warmup_epochs + 1

        loss, batch_time = pretrain(model, pretrain_loader, optimizer, criterion, local_args, epoch)

        if epoch >= local_args.box.warmup_epochs and epoch != local_args.run_p.pretrain_epoch and epoch % local_args.box.loc_interval == 0:
            # all_boxes: tensor (len_ds, 4); (h_min, w_min, h_max, w_max)
            all_boxes = update_box(eval_loader, model.module.encoder_q, len_ds,
                                   True if local_args.local_rank == 0 else False,
                                   t=local_args.box.box_thresh)  # on_cuda=True
            assert len(all_boxes) == len_ds
            pretrain_set.boxes = all_boxes.cpu()

        if local_args.local_rank == 0:
            dm.end_pretrain_epoch(model.module, loss, batch_time)
