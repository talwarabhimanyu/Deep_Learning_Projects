### CIFAR-10 Experiments
I intend to run experiments on the CIFAR-10 dataset, including but not limited to:
* Impact of learning rate regimes on convergence in loss.
* Impact of network architectures on convergence in loss.
* Impact of regularization schemes on convergence in loss.
* Impact of data augmentation regimes on convergence in loss.

I am writing my own wrapper nameed 'tau' which sits on top of PyTorch and allows me to conduct my experiments with ease. Some of its functionality includes:

#### Learning Rate Exploration
Tau allows the user to train a prediction function (such as a Neural Net) for a specified number of iterations under a learning rate ('lr') regime in which the 'lr' increases from a specified 'min_lr' to a 'max_lr' in a log-linear manner. It then plots smoothed training loss versus learning rates (refer plot below). This plot helps in determining the maximum learning rate which may be used for training. It also allows the user to zoom into a particular section of the plot (see Figure 2 below).

**Figure 1: Smoothed Loss vs Learning Rate Plot Produced by Tau**

![lr_curve](https://github.com/talwarabhimanyu/Deep_Learning_Projects/blob/master/Vision/data/lr_curve.png)

**Figure 2: Zoomed-in View of a Section of the LR Plot from Figure 1**

![lr_curve_zoom](https://github.com/talwarabhimanyu/Deep_Learning_Projects/blob/master/Vision/data/lr_curve_zoom.png)

#### Weight Visualization
Tau allows visualizing evolution of weights learnt by filters in the first convolutional layer. An example GIF file created by Tau can be seen below:

<img src="https://github.com/talwarabhimanyu/Deep_Learning_Projects/blob/master/Vision/data/weights.gif" width="500" height="500"/>

### CNN-Speed-Up
A comparison of my implementations (in Python) of forward/backward passes through a Convolution Layer. Some highlights:
* My fully vectorized forward pass is **>3x faster** than my naive-most implementation (which utilizes Python for loops).
* The Cythonized version of my naive-most forward pass is **~1.25x faster** than my naive-most forward pass in Python.

