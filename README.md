# Comparison of Continual Learning Methods
Continual Learning (CL) is a machine learning paradigm that focuses on training models to learn continuously from a stream of data, without forgetting previously learned knowledge. In the context of image classification tasks, CL involves training models to sequentially learn from multiple datasets or tasks over time, adapting to new data while retaining the ability to perform well on previously encountered tasks.

## Implementation

The datasets are strategically partitioned into multiple tasks, each representing a distinct subset of the data or a unique classification challenge. The number of tasks can be customized based on user preferences. During training, the neural network is sequentially exposed to each task, learning from the specific subset of data associated with it. The testing phase involves evaluating the model's performance not only on the current task but also on previously encountered tasks. This iterative process facilitates the creation of a model with consistent performance across all tasks, thus ensuring its suitability for diverse domains and novel classes.

## Continual Learning Methods

1. **Fine-tuning**: It involves training models sequentially, one task after the other. The model is initially trained on the first task and then fine-tuned on subsequent tasks, adjusting its parameters to learn new patterns while preserving knowledge from previous tasks. This technique is to know the lower-bound performance of the neural network across multiple tasks.

2. **Joint-datasets**: In this approach, all datasets are combined into a single training set, and the model is trained jointly on all tasks simultaneously. This method aims to leverage the diversity of the datasets to improve generalization and adaptability. This technique is to know the upper-bound performance of the neural network across multiple tasks.

3. **Elastic Weight Consolidation (EWC)**: EWC is a regularization technique that mitigates catastrophic forgetting by preserving important parameters learned during previous tasks. It achieves this by penalizing changes to critical weights based on their importance for previous tasks.

4. **Learning without Forgetting (LwF)**: LwF addresses forgetting by distilling knowledge from the previous model onto the current model during training on new tasks. It does so by using the previous model's predictions as soft targets to guide the learning process. Additionally, an alternative training approach involves incorporating an auxiliary network that is optimized for the current task [1]. This results in a loss function comprising both a stability term (based on the previous network) and a plasticity term (related to the auxiliary network).

5. **Bilateral Memory Consolidation (BiMeCo)**: BiMeCo incorporates two neural networks: a short-term network and a long-term network. The short-term network is designed for rapid learning from new tasks, while the long-term network serves as a repository for storing essential information from previous tasks. The memory consolidation process involves knowledge distillation and feature extraction, facilitating the transfer of knowledge from the short-term network to the long-term network while minimizing interference with existing knowledge [2].

6. **BiMeCo + LwF**: This approach combines BiMeCo with LwF, leveraging the strengths of both methods to enhance performance and mitigate forgetting.

## Run the code

To initiate training using various continual learning methods and apply multiple techniques, please follow these instructions:

1. Clone this repository to your local machine.
  ```bash
  https://github.com/pascutc98/continual-learning-methods
  cd continual-learning-methods
  ```
2. Create and activate a conda environment:
  ```bash
  conda create -n cl_methods python=3.8
  conda activate cl_methods
  ```
3. Install the required dependencies by using the provided `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```
4. Execute the file ```run_main.sh``` or ```run main.py``` directly. You can modify the input parameters as needed:
  ```bash
  bash run_main.sh
  ```
  ```bash
  python main.py
  ```
## Input parameters

Here's detailed information about the input parameters:

- General Parameters:
    - exp_name: Name of the experiment or project.
    - seed: Random seed for reproducibility.
    - epochs: Number of training epochs.
    - lr: Learning rate for optimization.
    - lr_decay: Learning rate decay factor.
    - lr_patience: Number of epochs to wait before reducing the learning rate.
    - lr_min: Minimum learning rate threshold.
    - batch_size: Batch size for training.
    - num_tasks: Number of tasks in the continual learning setup.
      
- Dataset Parameters
    - dataset: Choice of dataset for experimentation (e.g., mnist, cifar10, cifar100, cifar100-alternative-dist).
      
- EWC Parameters
    - ewc_lambda: Regularization parameter for Elastic Weight Consolidation (EWC).
      
- Distillation Parameters (LwF)
    - lwf_lambda: Hyperparameter controlling the importance of distillation loss in Learning without Forgetting (LwF).
    - lwf_aux_lambda: Hyperparameter controlling the importance of auxiliary distillation loss in LwF.
      
- BiMeCo Parameters
    - memory_size: Size of the memory buffer in Bilateral Memory Consolidation (BiMeCo).
    - bimeco_lambda_short: Regularization parameter for short-term network in BiMeCo.
    - bimeco_lambda_long: Regularization parameter for long-term network in BiMeCo.
    - bimeco_lambda_diff: Regularization parameter controlling the difference between the feature extractors of short-term and long-term networks in BiMeCo.
    - m: Momentum parameter for updating the model parameters.

Understanding these parameters will allow you to customize the training process and experiment with different configurations to achieve optimal results. For more information about these parameters, you can run the following command: 
  ```
  python main.py --help
  ```

## Results

We meticulously document the outcomes of each CL method in an Excel file for comprehensive evaluation. The results showcase the effectiveness of each technique in adapting to new tasks while maintaining performance on previous tasks. Through careful analysis, we provide insights into the strengths and limitations of different CL approaches in the context of image classification tasks.

## References
[1] Sanghwan Kim, Lorenzo Noci, Antonio Orvieto, Thomas Hofmann. [Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning](https://arxiv.org/abs/2303.09483). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11930-11939. 2023.

[2] Xing Nie, Shixiong Xu, Xiyan Liu, Gaofeng Meng, Chunlei Huo, Shiming Xiang. [Bilateral Memory Consolidation for Continual Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Nie_Bilateral_Memory_Consolidation_for_Continual_Learning_CVPR_2023_paper.html). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16026-16035.








