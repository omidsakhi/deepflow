#include "nodes/transposed_conv_2d.h"


TransposedConvolution2D::TransposedConvolution2D(const deepflow::NodeParam &param) : Node(param) {
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
	auto input_dims = _inputs[0]->dims();
	auto filterDims = _inputs[1]->dims();
	int input_n = input_dims[0];
	int input_c = input_dims[1];
	int input_h = input_dims[2];
	int input_w = input_dims[3];
	int filter_n = filterDims[0];
	int filter_c = filterDims[1];
	int filter_h = filterDims[2];
	int filter_w = filterDims[3];
	int output_n = input_n;
	int output_c = filter_c;
	int output_h = (input_h - 1) * u - 2 * pad_h + ((filter_h - 1) * dilation_h) + 1;
	int output_w = (input_w - 1) * v - 2 * pad_w + ((filter_w - 1) * dilation_w) + 1;
	_outputs[0]->initValue({ output_n, output_c, output_h, output_w });
	_outputs[0]->initDiff();	
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));		
	DF_NODE_CUDNN_CHECK(cudnnCreateFilterDescriptor(&_wDesc));	
	auto filterValueTensor = _inputs[1]->value();	
	DF_NODE_CUDNN_CHECK(cudnnSetFilter4dDescriptor(_wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_n, filter_c, filter_h, filter_w));
	DF_NODE_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&_convDesc));
	DF_NODE_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	int n, c, h, w;	
	DF_NODE_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(_convDesc, _outputs[0]->value()->descriptor(), _wDesc, &n, &c, &h, &w));	
	LOG(INFO) << "Initializing Transposed Convolution " << _name << " - " << _inputs[0]->value()->shape() << " * " << _inputs[1]->value()->shape() << " -> " << _outputs[0]->value()->shape();
	LOG_IF(FATAL, n != input_n || c != input_c || h != input_h || w != input_w) << "Input shape must be " << n << "*" << c << "*" << h << "*" << w;			
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _outputs[0]->value()->descriptor(), _wDesc, _convDesc, _inputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _outputs[0]->value()->descriptor(), _wDesc, _convDesc, _inputs[0]->value()->descriptor(), _fwdAlgo, &_fwdWorkspaceSize));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _wDesc, _inputs[0]->value()->descriptor(), _convDesc, _outputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _wDesc, _inputs[0]->value()->descriptor(), _convDesc, _outputs[0]->value()->descriptor(), _bwdDataAlgo, &_bwdDataWorkspaceSize));
	if (_inputs[0]->diff()) {
		_dy = (float*)_inputs[0]->diff()->mutableData();
	}
	else {
		cudaMalloc(&_dy, _inputs[0]->value()->sizeInBytes());
	}
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(_cudnnHandle, _outputs[0]->value()->descriptor(), _inputs[0]->value()->descriptor(), _convDesc, _wDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_bwdFilterAlgo));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(_cudnnHandle, _outputs[0]->value()->descriptor(), _inputs[0]->value()->descriptor(), _convDesc, _wDesc, _bwdFilterAlgo, &_bwdFilterWorkspaceSize));
	_maxWorkspaceSize = std::max({ _fwdWorkspaceSize, _bwdDataWorkspaceSize, _bwdFilterWorkspaceSize });
}

void TransposedConvolution2D::initBackward() {
}

void TransposedConvolution2D::forward() {
	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		DF_NODE_CUDA_CHECK(cudaMalloc(&d_workspace, _maxWorkspaceSize));		
	DF_NODE_CUDNN_CHECK(cudnnConvolutionBackwardData(_cudnnHandle, &one, _wDesc, _inputs[1]->value()->data(), _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _convDesc, _bwdDataAlgo, d_workspace, _bwdDataWorkspaceSize, &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));
}

void TransposedConvolution2D::backward() {
	if (_inputs[0]->connectedNode()->propagateBack())
		DF_NODE_CUDNN_CHECK(cudnnConvolutionForward(_cudnnHandle, &one, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _wDesc, _inputs[1]->value()->data(), _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &one, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));	
	if (_inputs[1]->connectedNode()->propagateBack())
		DF_NODE_CUDNN_CHECK(cudnnConvolutionBackwardFilter(_cudnnHandle, &one, _outputs[0]->value()->descriptor(), _outputs[0]->value()->data(), _inputs[0]->value()->descriptor(), _dy, _convDesc, _bwdFilterAlgo, d_workspace, _bwdFilterWorkspaceSize, &one, _wDesc, _inputs[1]->diff()->mutableData()));
}

std::string TransposedConvolution2D::to_cpp() const
{	
	auto param = _param.transposed_conv_2d_param();
	std::string cpp = "auto " + _name + " = df.transposed_conv2d(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	auto dims = _outputs[0]->value()->dims();
	cpp += "{" + std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ", " + std::to_string(dims[2]) + ", " + std::to_string(dims[3]) + "}";
	cpp += std::to_string(param.pad_h()) + ", ";
	cpp += std::to_string(param.pad_w()) + ", ";
	cpp += std::to_string(param.u()) + ", ";
	cpp += std::to_string(param.v()) + ", ";
	cpp += std::to_string(param.dilation_h()) + ", ";
	cpp += std::to_string(param.dilation_w()) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
