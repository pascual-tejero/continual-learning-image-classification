def on_task_update(task_id, model, optimizer, x_mem, t_mem):


  model.train() # Set the model in training mode
  optimizer.zero_grad() # Initialize gradients to zero

  # Loop through the training data in batches of size 256
  for start in range(0, len(t_mem)-1, 256):
      end = start + 256 # It calculates the end index of the current batch

      # Convert the batches of training data into Pytorch tensors
      x, y = torch.from_numpy(x_mem[start:end]), torch.from_numpy(t_mem[start:end]).long()

      # It moves the tensors to the specified computing device (e.g., GPU or CPU)
      x, y = x.to(device), y.to(device)

      # Make predictions
      output = model(x)

      # Compute cross-entropy loss
      loss = F.cross_entropy(output, y)

      # Performs backpropagation
      loss.backward()

  # Initialize dictionaries for Fisher information and model parameters
  fisher_dict[task_id] = {}
  optpar_dict[task_id] = {}

  # Calculate Fisher information for model parameters
  for name, param in model.named_parameters():

    # Store a clone of the model's parameter data
    optpar_dict[task_id][name] = param.data.clone()

    # Calculate Fisher information (squared gradients) for each parameter
    fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


def train_ewc(model, device, task_id, x_train, t_train, optimizer, epoch):
   

    model.train() # Set the model in training mode

    # Loop through the training data in batches of size 256
    for start in range(0, len(t_train)-1, 256):
      end = start + 256 # It calculates the end index of the current batch

      # Convert the batches of training data into Pytorch tensors
      x, y = torch.from_numpy(x_train[start:end]), torch.from_numpy(t_train[start:end]).long()

      # It moves the tensors to the specified computing device (e.g., GPU or CPU)
      x, y = x.to(device), y.to(device)

      # Make predictions
      optimizer.zero_grad()

      # Compute cross-entropy loss
      output = model(x)

      # Performs backpropagation
      loss = F.cross_entropy(output, y)


      ### magic here! :-)
      for task in range(task_id):
        for name, param in model.named_parameters():
          # Fisher information for the parameter
          fisher = fisher_dict[task][name]

          # Previous parameter value
          optpar = optpar_dict[task][name]

          # EWC loss is added to the original loss to prevent catastrophic forgetting.
          # It penalizes changes in model parameters based on the Fisher information
          # and previous parameter values
          loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

      loss.backward()
      optimizer.step()
      #print(loss.item())
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

fisher_dict = {}
optpar_dict = {}
ewc_lambda = 0.4

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ewc_accs = []  # Initialize a list to store average accuracy values for each task
for id, task in enumerate(tasks):
    avg_acc = 0  # Initialize average accuracy for the current task
    print("Training on task: ", id)  # Print the task being trained

    (x_train, t_train), _ = task  # Unpack training data for the current task

    for epoch in range(1, 4):  # Iterate through training epochs (typically 2 epochs)
        # Train the model for the current task
        train_ewc(model, device, id, x_train, t_train, optimizer, epoch)

    on_task_update(id, x_train, t_train)  # Update the model with task-specific information

    for id_test, task in enumerate(tasks):
        print("Testing on task: ", id_test)  # Print the task being tested
        _, (x_test, t_test) = task  # Unpack testing data for the current task
        acc = test(model, device, x_test, t_test)  # Test the model on the current task
        avg_acc = avg_acc + acc  # Accumulate accuracy across tasks

    print("Avg acc: ", avg_acc / 3)  # Calculate and print the average accuracy for the tasks
    ewc_accs.append(avg_acc / 3)  # Append the average accuracy to the list
