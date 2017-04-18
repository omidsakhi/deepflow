#include "ops/transposed_conv_2d.h"


TransposedConvolution2D::TransposedConvolution2D(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_transposed_conv_2d_param() == false) << "param.has_transposed_conv_2d_param() [FAILED]";
	d_workspace = 0;
}

void TransposedConvolution2D::initForward() {	
	auto transposed_conv2d_param = _param.transposed_conv_2d_param();
	int pad_h = transposed_conv2d_param.pad_h();
	int pad_w = transposed_conv2d_param.pad_w();
	int u = transposed_conv2d_param.u();
	int v = transposed_conv2d_param.v();
	int dilation_h = transposed_conv2d_param.dilation_h();
	int dilation_w = transposed_conv2d_param.dilation_w();
	LOG_IF(FATAL, pad_h < 0) << "pad_top_bottom (pad_h) < 0";
	LOG_IF(FATAL, pad_w < 0) << "pad_left_right (pad_w) < 0";
	LOG_IF(FATAL, u <= 0) << "vertical_filter_stride (u) <= 0";
	LOG_IF(FATAL, v <= 0) << "horizontal_filter_stride (v) <= 0";
	LOG_IF(FATAL, dilation_h <= 0) << "filter_height_dilation (dilation_h) <= 0";
	LOG_IF(FATAL, dilation_w <= 0) << "filter_width_dilation (dilation_w) <= 0";
	auto tensor_param = transposed_conv2d_param.tensor_param();
	int output_n = tensor_param.dims(0);
	int output_c = tensor_param.dims(1);
	int output_h = tensor_param.dims(2);
	int output_w = tensor_param.dims(3);
	_outputs[0]->initValue({ output_n, output_c, output_h, output_w });
	_outputs[0]->initDiff();
	LOG_IF(FATAL, tensor_param.dims_size() != 4) << "tensor_param.dims_size() != 4";	
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_CUDNN_CHECK(cudnnCreateFilterDescriptor(&_filterDesc));
	auto filterDims = _inputs[1]->dims();
	auto filterValueTensor = _inputs[1]->value();	
	DF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(_filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3]));
	DF_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&_convDesc));
	DF_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	int input_n = _inputs[0]->value()->dim(0);
	int input_c = _inputs[0]->value()->dim(1);
	int input_h = _inputs[0]->value()->dim(2);
	int input_w = _inputs[0]->value()->dim(3);
	auto input_dims = _inputs[0]->dims();
	int n, c, h, w;	
	DF_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(_convDesc, _outputs[0]->value()->descriptor(), _filterDesc, &n, &c, &h, &w));	
	LOG_IF(FATAL, n != input_n || c != input_c || h != input_h || w != input_w) << "Input shape must be " << n << "*" << c << "*" << h << "*" << w;
	LOG_IF(FATAL, filterDims[1] != c) << "Output channels " << output_c << " != Filter channels " << filterDims[1];

	LOG(INFO) << "Initializing Transposed Convolution " << _name << " - " << _inputs[0]->value()->shape() << " * " << _inputs[1]->value()->shape() << " -> " << _outputs[0]->value()->shape();
	
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _outputs[0]->value()->descriptor(), _filterDesc, _convDesc, _inputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _outputs[0]->value()->descriptor(), _filterDesc, _convDesc, _inputs[0]->value()->descriptor(), _fwdAlgo, &_fwdWorkspaceSize));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _filterDesc, _inputs[0]->value()->descriptor(), _convDesc, _outputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _filterDesc, _inputs[0]->value()->descriptor(), _convDesc, _outputs[0]->value()->descriptor(), _bwdDataAlgo, &_bwdDataWorkspaceSize));

	_maxWorkspaceSize = std::max({ _fwdWorkspaceSize, _bwdDataWorkspaceSize });
}

void TransposedConvolution2D::initBackward() {
}

void TransposedConvolution2D::forward() {
	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		DF_CUDA_CHECK(cudaMalloc(&d_workspace, _maxWorkspaceSize));		
	DF_CUDNN_CHECK(cudnnConvolutionBackwardData(_cudnnHandle, &alpha, _filterDesc, _inputs[1]->value()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->diff()->data(), _convDesc, _bwdDataAlgo, d_workspace, _bwdDataWorkspaceSize, &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void TransposedConvolution2D::backward() {
	DF_CUDNN_CHECK(cudnnConvolutionForward(_cudnnHandle, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _filterDesc, _inputs[1]->value()->data(), _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}
