### CIFAR-10 Experiments
I intend to run experiments on the CIFAR-10 dataset, including but not limited to:
* Impact of learning rate regimes on convergence in loss.
* Impact of network architectures on convergence in loss.
* Impact of regularization schemes on convergence in loss.
* Impact of data augmentation regimes on convergence in loss.

I am writing my own wrapper called 'tau' on top of PyTorch to conduct my experiments.

### CNN-Speed-Up
A comparison of my implementations (in Python) of forward/backward passes through a Convolution Layer. Some highlights:
* My fully vectorized forward pass is **>3x faster** than my naive-most implementation (which utilizes Python for loops).
* The Cythonized version of my naive-most forward pass is **~1.25x faster** than my naive-most forward pass in Python.

