#!/usr/bin/env bash
python train.py \
    --dataroot_A "/media/data_cifs/sven2/leaves/sorted/Leaf_Splits/Split_S1/Leaves_768/50_Training_data" \
    --dataroot_B "/media/data_cifs/sven2/leaves/sorted/Fossil_Splits/Fossil_Split_S1/Fossils_768/20_Training_data" \
    --dataset_A "leaves" \
    --dataset_B "fossils" \
    --name "leaves_to_fossils_cycleGAN" \
    --model "cycle_gan" \
    --netG "unet_256" \
    --checkpoints_dir "./output" \
    --batch_size 1 \
    --load_size 286 \
    --crop_size 256 \
    --dataset_mode "aligned" \
    --save_epoch_freq 40 \
    "$@"


