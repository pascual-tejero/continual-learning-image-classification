
import argparse

def analyse_datasets(datasets, args):
    """
    Count number of images per class in the dataset.
    """
    print("=" * 130)
    print("Number of tasks: ", len(datasets))
    print("Number of samples per task: ", [len(task[0]) for task in datasets])
    print("=" * 130)

    # Load the dataset
    for idx, (train_set, val_set, test_set) in enumerate(datasets):

        # Split each tensor dataset into images and labels
        x_train, t_train = train_set.tensors
        x_val, t_val = val_set.tensors
        x_test, t_test = test_set.tensors

        # Convert the tensors to numpy arrays
        x_train = x_train.numpy()
        t_train = t_train.numpy()
        x_val = x_val.numpy()
        t_val = t_val.numpy()
        x_test = x_test.numpy()
        t_test = t_test.numpy()
        

        # Get the number of images per class
        count_train = {}
        count_val = {}
        count_test = {}

        # Count the number of images per class in the sets
        for i in range(len(t_train)):
            if t_train[i] in count_train:
                count_train[t_train[i]] += 1
            else:
                count_train[t_train[i]] = 1

        for i in range(len(t_val)):
            if t_val[i] in count_val:
                count_val[t_val[i]] += 1
            else:
                count_val[t_val[i]] = 1

        for i in range(len(t_test)):
            if t_test[i] in count_test:
                count_test[t_test[i]] += 1
            else:
                count_test[t_test[i]] = 1

        # Sort dictionary by key
        count_train = dict(sorted(count_train.items()))
        count_val = dict(sorted(count_val.items()))
        count_test = dict(sorted(count_test.items()))

        # Count the total number of images per set
        count_train_num = sum(count_train.values())
        count_val_num = sum(count_val.values())
        count_test_num = sum(count_test.values())

        # Print the results in the terminal
        print(f"TASK {idx+1}: \n")
        print(f" - Train set {idx+1} -> TOTAL: {count_train_num} /// {count_train} \n")
        print(f" - Val set {idx+1} -> TOTAL: {count_val_num} ///  {count_val} \n")
        print(f" - Test set {idx+1} -> TOTAL: {count_test_num} /// {count_test} \n")
        print("=" * 130)

        # Save the results in a .txt file
        # with open(f'./results/{args.exp_name}/args_{args.exp_name}_{args.dataset}.txt', 'w') as f:
        #     f.write(f"TASK {idx+1}: \n")
        #     f.write(f" - Train set {idx+1} -> TOTAL: {count_train_num} /// {count_train} \n")
        #     f.write(f" - Val set {idx+1} -> TOTAL: {count_val_num} ///  {count_val} \n")
        #     f.write(f" - Test set {idx+1} -> TOTAL: {count_test_num} /// {count_test} \n")
        #     f.write("=" * 130)
        #     f.write("\n")
    


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--dataset', type=str, default='mnist')
    analyse_datasets(argparse.parse_args())
