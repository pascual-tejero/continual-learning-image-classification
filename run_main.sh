python main.py --exp_name mnist_training --seed 0 --epochs 500 --lr 0.0001 \
    --lr_decay 5 --lr_patience 10 --lr_min 1e-8 --batch_size 200 --num_tasks 2 \
    --dataset mnist --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 0.5

python main.py --exp_name cifar10_training --seed 0 --epochs 500 --lr 0.0001 \
    --lr_decay 5 --lr_patience 10 --lr_min 1e-8 --batch_size 200 --num_tasks 2 \
    --dataset cifar10 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 0.5

python main.py --exp_name cifar100_training --seed 0 --epochs 500 --lr 0.0001 \
    --lr_decay 5 --lr_patience 10 --lr_min 1e-8 --batch_size 200 --num_tasks 2 \
    --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 0.5


# python main.py --exp_name cifar100_lwf_lambda_aux_1 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 1

# python main.py --exp_name cifar100_lwf_lambda_aux_0.5 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 0.5

# python main.py --exp_name cifar100_lwf_lambda_aux_0.1 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 0.1

# python main.py --exp_name cifar100_lwf_lambda_aux_2 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 2

# python main.py --exp_name cifar100_lwf_lambda_aux_5 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 5

# python main.py --exp_name cifar100_lwf_lambda_aux_10 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 10

# python main.py --exp_name cifar100_lwf_lambda_aux_20 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 20

# python main.py --exp_name cifar100_lwf_lambda_aux_50 --seed 0 --epochs 200 --lr 0.001 \
#     --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
#     --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1 --lwf_aux_lambda 50