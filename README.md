# How to Start project?

1. Download outputs zip file from here and extract root project folder. There should be 3 folder their names "outputs_lr_0.1", "outputs_lr_0.01" and "outputs_lr_0.001" It contains pretrained models for different configuration.
2. Download basic_requirements.(requirements.txt is my working enviroment package list)


- There are 3 pyton source file is used in this task.

  1. cifar_DataCreate.py:it reads cifar10 batches and generate train, validation and test datasets.
  2. Dataset.py: it is used for Dataloader. It holds data Ids and labels.
  3. transfer_learning.py: it is used for train and test network with desired settings.
