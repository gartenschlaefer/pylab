# --
# laod mnist dataset

import torch
import torchvision

import matplotlib.pyplot as plt



if __name__ == '__main__':
  """
  main
  """

  # load data
  mnist_data = torchvision.datasets.MNIST(root='/world/mapletree/datasets/mnist', train=True, transform=None, target_transform=None, download=True)

  for x, y in mnist_data:
    print("y: ", y)