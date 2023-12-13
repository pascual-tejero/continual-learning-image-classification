python main.py --exp_name mnist_training --seed 0 --epochs 200 --lr 0.001 \
    --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
    --dataset mnist --ewc_lambda 1000 --lwf_lambda 1

python main.py --exp_name cifar10_training --seed 0 --epochs 200 --lr 0.001 \
    --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
    --dataset cifar10 --ewc_lambda 1000 --lwf_lambda 1

python main.py --exp_name cifar100_training --seed 0 --epochs 200 --lr 0.001 \
    --lr_decay 5 --lr_patience 5 --lr_min 1e-6 --batch_size 200 --num_tasks 2 \
    --dataset cifar100 --ewc_lambda 1000 --lwf_lambda 1

# # General parameters
# argparse.add_argument('--exp_name', type=str, default="exp")
# argparse.add_argument('--seed', type=int, default=0)
# argparse.add_argument('--epochs', type=int, default=1)
# argparse.add_argument('--lr', type=float, default=0.1)
# argparse.add_argument('--lr_decay', type=float, default=5)
# argparse.add_argument('--lr_patience', type=int, default=5)
# argparse.add_argument('--lr_min', type=float, default=1e-6)
# argparse.add_argument('--batch_size', type=int, default=200)
# argparse.add_argument('--num_tasks', type=int, default=2)
# argparse.add_argument('--scheduler_step_size', type=int, default=7)
# argparse.add_argument('--scheduler_gamma', type=float, default=0.3)

# # Dataset parameters: mnist, cifar10, cifar100
# argparse.add_argument('--dataset', type=str, default="cifar100")

# # EWC parameters
# argparse.add_argument('--ewc_lambda' , type=float, default=1000)

# # Distillation parameters
# argparse.add_argument('--lwf_lambda' , type=float, default=1)
