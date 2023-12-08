python ../main.py --seed 0 --dataset mnist --epochs 10 --lr 0.1 --batch-size 20 \
    --num_tasks 4 --dataset cifar10 --ewc_lambda 5 --lwf_lambda 5

python ../main.py --seed 0 --dataset cifar10 --epochs 10 --lr 0.1 --batch-size 20 \
    --num_tasks 4 --dataset cifar10 --ewc_lambda 5 --lwf_lambda 5

python ../main.py --seed 0 --dataset cifar100 --epochs 10 --lr 0.1 --batch-size 20 \
    --num_tasks 4 --dataset cifar10 --ewc_lambda 5 --lwf_lambda 5