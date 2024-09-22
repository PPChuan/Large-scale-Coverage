Requirements:
    python >= 3.9.16
    pytorch >= 2.0.0
    diffdist >= 0.1
    tensorboard >= 2.11.0

Function L (LSC) is implemented in ./build/datasets/transform/Covering.py/StepDeCovering

run demo(linux):
    python moco_rcrop.py --cfg=moco/cifar100_rcrop --i=false

    run demo introduction:
        run_files list:
            moco_rcrop.py (default moco)
            moco_rcrop_lsc.py (moco + LSC)

            moco_ccrop.py (moco + C-Crop)
            moco_ccrop_lsc.py (moco + C-Crop + LSC)

            moco_rcrop_amp_in1000.py (default moco with bfloat16 mixed precision, used on ImageNet 1k)
            moco_rcrop_lsc_amp_in1000.py (moco + LSC with bfloat16 mixed precision, used on ImageNet 1k)

            simclr_rcrop.py (default simclr)
            simclr_rcrop_lsc.py (simclr + LSC)

            simclr_ccrop.py (simclr + C-Crop)
            simclr_ccrop_lsc.py (simclr + C-Crop + LSC)

            byol_rcrop.py (default byol)
            byol_rcrop_lsc.py (byol + LSC)

            simsiam_rcrop.py (default simsiam)
            simsiam_rcrop_lsc.py (simsiam + LSC)

        config dir: ./cfg/*
            !!!experiments involving cutout use the default model run_files like
            python moco_rcrop.py --cfg=moco/cifar100_rcrop_cutout --i=false

    config introduction: look in ./cfg/moco/intro.yml