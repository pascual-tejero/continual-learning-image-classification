python main.py --exp_name mnist_training --seed 0 --epochs 500 --lr 0.001 --lr_decay 5 --lr_patience 10 \
    --lr_min 1e-6 --batch_size 128 --num_tasks 2 --dataset mnist --img_size 28 --img_channels 1 --num_classes 10 \
    --feature_dim 320 --ewc_lambda 1000 --lwf_lambda 0.80 --lwf_aux_lambda 0.76 --memory_size 250000 \
    --bimeco_lambda_long 2.5 --bimeco_lambda_short 1.5 --bimeco_lambda_diff 4 --m 0.15 

python main.py --exp_name cifar10_training --seed 0 --epochs 500 --lr 0.001 --lr_decay 5 --lr_patience 10 \
    --lr_min 1e-6 --batch_size 128 --num_tasks 2 --dataset cifar10 --img_size 32 --img_channels 3 --num_classes 10 \
    --feature_dim 512 --ewc_lambda 1000 --lwf_lambda 0.80 --lwf_aux_lambda 0.76 --memory_size 250000 \
    --bimeco_lambda_long 2.5 --bimeco_lambda_short 1.5 --bimeco_lambda_diff 4 --m 0.15

python main.py --exp_name cifar100_training --seed 0 --epochs 500 --lr 0.001 --lr_decay 5 --lr_patience 10 \
    --lr_min 1e-6 --batch_size 128 --num_tasks 2 --dataset cifar100 --img_size 32 --img_channels 3 --num_classes 100 \
    --feature_dim 64 --ewc_lambda 1000 --lwf_lambda 0.80 --lwf_aux_lambda 0.76 --memory_size 250000 \
    --bimeco_lambda_long 2.5 --bimeco_lambda_short 1.5 --bimeco_lambda_diff 4 --m 0.15