# Comparison of Continual Learning Methods
Continual Learning (CL) is a machine learning paradigm that focuses on training models to learn continuously from a stream of data, without forgetting previously learned knowledge. In the context of image classification tasks, CL involves training models to sequentially learn from multiple datasets or tasks over time, adapting to new data while retaining the ability to perform well on previously encountered tasks.

## Implementation

The datasets are strategically partitioned into multiple tasks, each representing a distinct subset of the data or a unique classification challenge. The number of tasks can be customized based on user preferences. During training, the neural network is sequentially exposed to each task, learning from the specific subset of data associated with it. The testing phase involves evaluating the model's performance not only on the current task but also on previously encountered tasks. This iterative process facilitates the creation of a model with consistent performance across all tasks, thus ensuring its suitability for diverse domains and novel classes.

## Continual Learning Methods

1. **Fine-tuning**: It involves training models sequentially, one task after the other. The model is initially trained on the first task and then fine-tuned on subsequent tasks, adjusting its parameters to learn new patterns while preserving knowledge from previous tasks. This technique is to know the lower-bound performance of the neural network across multiple tasks.

2. **Joint-datasets**: In this approach, all datasets are combined into a single training set, and the model is trained jointly on all tasks simultaneously. This method aims to leverage the diversity of the datasets to improve generalization and adaptability. This technique is to know the upper-bound performance of the neural network across multiple tasks.

3. **Elastic Weight Consolidation (EWC)**: EWC is a regularization technique that mitigates catastrophic forgetting by preserving important parameters learned during previous tasks. It achieves this by penalizing changes to critical weights based on their importance for previous tasks.

4. **Learning without Forgetting (LwF)**: LwF addresses forgetting by distilling knowledge from the previous model onto the current model during training on new tasks. It does so by using the previous model's predictions as soft targets to guide the learning process.

5. **Bilateral Memory Consolidation (BiMeCo)**:

6. **BiMeCo + LwF**: This approach combines BiMeCo with LwF, leveraging the strengths of both methods to enhance performance and mitigate forgetting.

## Results

We meticulously document the outcomes of each CL method in an Excel file for comprehensive evaluation. The results showcase the effectiveness of each technique in adapting to new tasks while maintaining performance on previous tasks. Through careful analysis, we provide insights into the strengths and limitations of different CL approaches in the context of image classification tasks.





