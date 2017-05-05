# Deepflow

# Features

- Node base like Tensorflow
- Multiple execution phases per graph
- Custom solver per variable
- Live weight/data display
- Create C++ code from Deepflow model

# In Progress

- A method to convert Caffe models to Deeflow models

# Requirments

- NVIDIA Graphics Card
- GPU Only

# Dependencies

- Visual Studio 2015
- [CUDA 8](https://developer.nvidia.com/cuda-toolkit)
- [OpenCV 3.0](http://opencv.org/opencv-3-0.html)
- [cuDNN v6.0](https://developer.nvidia.com/rdp/cudnn-download)
- [Protocol Buffers](https://github.com/google/protobuf)
- [glog](https://github.com/google/glog)
- [gflags](https://github.com/gflags/gflags)

# Current Nodes

| Nodes                 |                       |                       |                       |
|-----------------------|:---------------------:|:---------------------:|----------------------:|
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
| cast_float            | accumulator           |                       |                       |

# Current Solvers

- AdaDelta
- Adam
- Gain
- Stochastic Gradient Descent
