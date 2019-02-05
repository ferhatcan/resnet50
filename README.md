# How to Start project?

1. Download outputs zip file from here and extract root project folder. There should be 3 folder their names __"outputs_lr_0.1", "outputs_lr_0.01" and "outputs_lr_0.001"__ It contains pretrained models for different configuration.
2. Download basic_requirements.(requirements.txt is my working enviroment package list)


- There are 3 pyton source file is used in this task.

  1. __cifar_DataCreate.py:__ It reads cifar10 batches and generate train, validation and test datasets.
  2. __Dataset.py:__ It is used for Dataloader. It holds data Ids and labels.
  3. __transfer_learning.py:__ It is used for train and test network with desired settings.

### Note: Transfer_learning_resnet50.ipynb includes testing steps with some results obtained.
