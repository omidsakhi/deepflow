# Deepflow
A node based deep neural network library on top of CuDNN

# Status
A work in progress. Not mature enough for a release but cool enough for poking around.

# Features
- Node base like Tensorflow
- Multiple execution phases per graph
- Custom solver per variable
- Live weight/data display
- Create C++ code from Deepflow model

# Requirments
- NVIDIA Graphics Card
- GPU Only

# Dependencies
- Visual Studio 2015
- [CUDA 9.0](https://developer.nvidia.com/cuda-toolkit)
- [OpenCV 3.0](http://opencv.org/opencv-3-0.html)
- [cuDNN v7.1](https://developer.nvidia.com/rdp/cudnn-download)
- [Protocol Buffers](https://github.com/google/protobuf)
- [glog](https://github.com/google/glog)
- [gflags](https://github.com/gflags/gflags)

# Working Examples
- face_dcgan: A least-square generative adversarial network.

![results](https://github.com/omidsakhi/deepflow/blob/master/examples/face_dcgan/results.jpg)

- face_ac: Deep auto-encoder on 128x128 CelebA dataset.

![results](https://github.com/omidsakhi/deepflow/blob/master/examples/face_ac/results.jpg)

- face_vae: Deep variaional auto-encoder on 128x128 CelebA dataset.

![results](https://github.com/omidsakhi/deepflow/blob/master/examples/face_vae/results.jpg)

- mnist_ac: Deep auto-encoder on MNIST dataset.

![results](https://github.com/omidsakhi/deepflow/blob/master/examples/mnist_ac/results.jpg)

- mnist_dcgan: Least-square gan (generative adversarial) on MNIST dataset.

![results](https://github.com/omidsakhi/deepflow/blob/master/examples/mnist_dcgan/results.jpg)

- mnist_lenet


# Current Nodes
| Nodes                 | Nodes                 | Nodes                 | Nodes                 |
|-----------------------|-----------------------|-----------------------|-----------------------|
| data_generator        | variable              | place_holder          | conv2d                |
| image_batch_generator | pooling               | convolution_2d        | transposed_conv2d     |
| image_reader          | add                   | square                | matmult               |
| mnist_reader          | subtract              | bias_add              | dropout               |
| argmax                | argmin                | reduce_max            | reduce_min            |
| reduce_mean           | reduce_sum            | reduce_absmax         | reduce_norm1          |
| reduce_norm2          | leaky_relu            | sigmoid               | relu                  |
| tanh                  | clipped_relu          | elu                   | phaseplexer           |
| random_selector       | softmax_loss          | euclidean_loss        | print                 |
| display               | psnr                  | softmax               | equal                 |
| cast_float            | accumulator           | batch_normalization   | logger                |
| negate                | multiplexer           | random_selector       | restructure           |
| softmax               | ...                   |                       |                       |

# Current Solvers
- AdaDelta
- Adam
- RMSProp
- SGD
