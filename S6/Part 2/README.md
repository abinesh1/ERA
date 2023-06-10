# S6 Solution

## Problem statement

Achieve 99.4% accuracy with lesser than 20000 parameters and 20 epochs

## Result

Highest accuracy: 99.49
No: of parameters: 18,914

![Result](./images/11%20epoch.png)

## Model highlights

 - Model has 3 conv blocks, 2 transition blocks and 1 output block
 - Max number of channels = 32
 - Batch size = 128
 - Receptive field = 32
 - Dropout rates of 0.1, 0.15, 0.2 used across blocks
 - Batch normalization used between layers
 - Used a Global average Pooling layer
 - Optimizer used is SGD with learning rate=0.01 and momentum=0.9
 - ReduceLROnPlateau is used as scheduler


## Network architecture

```c

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4           [-1, 16, 28, 28]           1,152
              ReLU-5           [-1, 16, 28, 28]               0
         MaxPool2d-6           [-1, 16, 14, 14]               0
           Dropout-7           [-1, 16, 14, 14]               0
       BatchNorm2d-8           [-1, 16, 14, 14]              32
            Conv2d-9           [-1, 16, 14, 14]           2,304
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
           Conv2d-12           [-1, 32, 14, 14]           4,608
             ReLU-13           [-1, 32, 14, 14]               0
        MaxPool2d-14             [-1, 32, 7, 7]               0
          Dropout-15             [-1, 32, 7, 7]               0
      BatchNorm2d-16             [-1, 32, 7, 7]              64
           Conv2d-17             [-1, 32, 7, 7]           9,216
             ReLU-18             [-1, 32, 7, 7]               0
        AvgPool2d-19             [-1, 32, 1, 1]               0
      BatchNorm2d-20             [-1, 32, 1, 1]              64
           Conv2d-21             [-1, 32, 1, 1]           1,024
             ReLU-22             [-1, 32, 1, 1]               0
...
Forward/backward pass size (MB): 0.64
Params size (MB): 0.07
Estimated Total Size (MB): 0.71
----------------------------------------------------------------

```

## Model

```

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False),       # In - 28x28x1, Out - 28x28x6, RF - 3x3
                                nn.ReLU(),
                                nn.BatchNorm2d(8),
                                nn.Conv2d(8, 16, 3, padding=1, bias=False),      # In - 28x28x6, Out - 28x28x12, RF - 5x5
                                nn.ReLU()
                              )

    self.transition1 = nn.Sequential(nn.MaxPool2d(2, 2))                        # In - 28x28x12, Out - 14x14x12, RF - 6x6

    self.conv2 = nn.Sequential(nn.Dropout(0.15),
                                nn.BatchNorm2d(16),
                                nn.Conv2d(16, 16, 3, padding=1, bias=False),     # In - 14x14x12, Out - 14x14x16, RF - 10x10
                                nn.ReLU(),
                                nn.BatchNorm2d(16),
                                nn.Conv2d(16, 32, 3, padding=1, bias=False),    # In - 14x14x16, Out - 14x14x16, RF - 14x14
                                nn.ReLU()
                              )

    self.transition2 = nn.Sequential(nn.MaxPool2d(2, 2))                        # In - 14x14x16, Out - 7x7x16, RF - 16x16

    self.conv3 = nn.Sequential(nn.Dropout(0.2),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(32, 32, 3, padding=1, bias=False),    # In - 7x7x16, Out - 7x7x16, RF - 24x24
                                nn.ReLU()
                              )

    self.final_layers = nn.Sequential(nn.AvgPool2d(7, 7),
                                      nn.BatchNorm2d(32),
                                      nn.Conv2d(32, 32, 1, bias=False),         # In - 7x7x16, Out - 7x7x32, RF - 32x32
                                      nn.ReLU(),
                                      nn.Conv2d(32, 10, 1),                     # In - 7x7x32, Out - 7x7x10, RF - 32x32
                                      nn.Flatten(),
                                      nn.LogSoftmax()
                                      )

  def forward(self, x):
      x = self.conv1(x)
      x = self.transition1(x)
      x = self.conv2(x)
      x = self.transition2(x)
      x = self.conv3(x)
      x = self.final_layers(x)
      return x

```