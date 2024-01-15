python main.py --exp_name "mnist_mem50%" --dataset mnist --memory_size 290250
python main.py --exp_name "mnist_mem30%" --dataset mnist --memory_size 174150
python main.py --exp_name "mnist_mem10%" --dataset mnist --memory_size 58050

python main.py --exp_name "cifar10_mem50%" --dataset cifar10 --memory_size 245250
python main.py --exp_name "cifar10_mem30%" --dataset cifar10 --memory_size 147150
python main.py --exp_name "cifar10_mem10%" --dataset cifar10 --memory_size 49050

python main.py --exp_name "cifar100_mem50%" --dataset cifar100 --memory_size 22500
python main.py --exp_name "cifar100_mem30%" --dataset cifar100 --memory_size 13500
python main.py --exp_name "cifar100_mem10%" --dataset cifar100 --memory_size 4500

python main.py --exp_name "cifar100_leakage_data_mem50%" --dataset cifar100_alternative_dist --memory_size 22500 
python main.py --exp_name "cifar100_leakage_data_mem30%" --dataset cifar100_alternative_dist --memory_size 13500
python main.py --exp_name "cifar100_leakage_data_mem10%" --dataset cifar100_alternative_dist --memory_size 4500
