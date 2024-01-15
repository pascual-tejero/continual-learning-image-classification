
# If number of tasks is 2:
# MNIST and Fashion-MNIST: 59900 samples per task 
python main.py --exp_name "mnist_mem50%" --dataset mnist --memory_size 29745 # 29745/10 = 2974 samples per class
python main.py --exp_name "mnist_mem30%" --dataset mnist --memory_size 17847 # 17847/10 = 1784 samples per class
python main.py --exp_name "mnist_mem10%" --dataset mnist --memory_size 5949 # 5949/10 = 595 samples per class

# CIFAR10: 24500 samples per task
python main.py --exp_name "cifar10_mem50%" --dataset cifar10 --memory_size 12263 # 12263/5 = 2452 samples per class
python main.py --exp_name "cifar10_mem30%" --dataset cifar10 --memory_size 7358 # 7358/5 = 1471 samples per class
python main.py --exp_name "cifar10_mem10%" --dataset cifar10 --memory_size 2452 # 2452/5 = 490 samples per class

# CIFAR100: 22500 samples per task
python main.py --exp_name "cifar100_mem50%" --dataset cifar100 --memory_size 22500 # 22500/100 = 225 samples per class
python main.py --exp_name "cifar100_mem30%" --dataset cifar100 --memory_size 13500 # 13500/100 = 135 samples per class
python main.py --exp_name "cifar100_mem10%" --dataset cifar100 --memory_size 4500 # 4500/100 = 45 samples per class

# CIFAR100 with alternative distribution: 36490 samples in task 1 and 8945 samples in task 2 (with leakage data)
python main.py --exp_name "cifar100_leakage_data_mem50%" --dataset cifar100_alternative_dist --memory_size 22500 # 22500/100 = 225 samples per class
python main.py --exp_name "cifar100_leakage_data_mem30%" --dataset cifar100_alternative_dist --memory_size 13500 # 13500/100 = 135 samples per class
python main.py --exp_name "cifar100_leakage_data_mem10%" --dataset cifar100_alternative_dist --memory_size 4500 # 4500/100 = 45 samples per class
