#include "nodes/convolution_2d.h"

Convolution2D::Convolution2D(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_conv_2d_param() == false) << "param.has_conv_2d_param() [FAILED]";	
	d_workspace = 0;
}

void Convolution2D::init() {
	_xDesc = _inputs[0]->value()->descriptor();
	
	auto inputDims = _inputs[0]->dims();	
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_NODE_CUDNN_CHECK(cudnnCreateFilterDescriptor(&_wDesc));	
	auto filterDims = _inputs[1]->dims();	
	LOG_IF(FATAL, filterDims[1] != inputDims[1]) << _name << " Input channels " << inputDims[1] << " != Filter channels " << filterDims[1];	
	DF_NODE_CUDNN_CHECK(cudnnSetFilter4dDescriptor(_wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3]));	
	deepflow::Conv2dParam *param = _param->mutable_conv_2d_param();
	DF_NODE_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&_convDesc));	
	LOG_IF(FATAL, param->pad_h() < 0) << "pad_top_bottom (pad_h) < 0";
	LOG_IF(FATAL, param->pad_w() < 0) << "pad_left_right (pad_w) < 0";
	LOG_IF(FATAL, param->u() <= 0) << "vertical_filter_stride (u) <= 0";
	LOG_IF(FATAL, param->v() <= 0) << "horizontal_filter_stride (v) <= 0";
	LOG_IF(FATAL, param->dilation_h() <= 0) << "filter_height_dilation (dilation_h) <= 0";
	LOG_IF(FATAL, param->dilation_w() <= 0) << "filter_width_dilation (dilation_w) <= 0";
	DF_NODE_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_convDesc, param->pad_h(), param->pad_w(), param->u(), param->v(), param->dilation_h(), param->dilation_w(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));	
	int n, c, h, w;
	DF_NODE_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(_convDesc, _xDesc, _wDesc, &n, &c, &h, &w));
	_outputs[0]->initValue({ n, c, h, w });
	_yDesc = _outputs[0]->value()->descriptor();	
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _xDesc, _wDesc, _convDesc, _yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _xDesc, _wDesc, _convDesc, _yDesc, _fwdAlgo, &_fwdWorkspaceSize));	
	_maxWorkspaceSize = _fwdWorkspaceSize;

	_outputs[0]->initDiff();
	_dyDesc = _outputs[0]->diff()->descriptor();	

	if (_inputs[0]->diff()) {
		_dxDesc = _inputs[0]->diff()->descriptor();
		DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _wDesc, _dyDesc, _convDesc, _dxDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo));
		DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _wDesc, _dyDesc, _convDesc, _dxDesc, _bwdDataAlgo, &_bwdDataWorkspaceSize));
		_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdDataWorkspaceSize });
	}
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(_cudnnHandle, _xDesc, _dyDesc, _convDesc, _wDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_bwdFilterAlgo));
	DF_NODE_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(_cudnnHandle, _xDesc, _dyDesc, _convDesc, _wDesc, _bwdFilterAlgo, &_bwdFilterWorkspaceSize));
	_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdFilterWorkspaceSize });

	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		DF_NODE_CUDA_CHECK(cudaMallocManaged(&d_workspace, _maxWorkspaceSize));
}

void Convolution2D::forward() {
	float *_x = _inputs[0]->value()->gpu_data();
	float *_w = _inputs[1]->value()->gpu_data();	
	float *_y = _outputs[0]->value()->gpu_data();
	DF_NODE_CUDNN_CHECK(cudnnConvolutionForward(_cudnnHandle, &one, _xDesc, _x, _wDesc, _w, _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &zero, _yDesc, _y));
}

void Convolution2D::backward() {	
	float *_dy = _outputs[0]->diff()->gpu_data();	
	if (_inputs[0]->diff()) {
		float *_w = _inputs[1]->value()->gpu_data();
		float *_dx = _inputs[0]->diff()->gpu_data();
		DF_NODE_CUDNN_CHECK(cudnnConvolutionBackwardData(_cudnnHandle, &one, _wDesc, _w, _dyDesc, _dy, _convDesc, _bwdDataAlgo, d_workspace, _bwdDataWorkspaceSize, &zero, _dxDesc, _dx));
	}
	if (_inputs[1]->diff()) {
		float *_x = _inputs[0]->value()->gpu_data();
		float *_dw = _inputs[1]->diff()->gpu_data();
		DF_NODE_CUDNN_CHECK(cudnnConvolutionBackwardFilter(_cudnnHandle, &one, _xDesc, _x, _dyDesc, _dy, _convDesc, _bwdFilterAlgo, d_workspace, _bwdFilterWorkspaceSize, &zero, _wDesc, _dw));
	}
}

std::string Convolution2D::to_cpp() const
{	
	const deepflow::Conv2dParam &param = _param->conv_2d_param();
	std::string cpp = "auto " + _name + " = df.conv2d(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += std::to_string(param.pad_h()) + ", ";
	cpp += std::to_string(param.pad_w()) + ", ";
	cpp += std::to_string(param.u()) + ", ";
	cpp += std::to_string(param.v()) + ", ";
	cpp += std::to_string(param.dilation_h()) + ", ";
	cpp += std::to_string(param.dilation_w()) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
