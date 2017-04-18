#include "ops/convolution_2d.h"

Convolution2D::Convolution2D(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_conv_2d_param() == false) << "param.has_conv_2d_param() [FAILED]";
	d_workspace = 0;
}

void Convolution2D::initForward() {	
	auto inputDims = _inputs[0]->dims();	
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_CUDNN_CHECK(cudnnCreateFilterDescriptor(&_filterDesc));	
	auto filterDims = _inputs[1]->dims();
	auto filterValueTensor = _inputs[1]->value();
	LOG_IF(FATAL, filterDims[1] != inputDims[1]) << "Input channels " << inputDims[1] << " != Filter channels " << filterDims[1];
	DF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(_filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3]));	
	Conv2dParam *param = _param.mutable_conv_2d_param();
	DF_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&_convDesc));	
	LOG_IF(FATAL, param->pad_h() < 0) << "pad_top_bottom (pad_h) < 0";
	LOG_IF(FATAL, param->pad_w() < 0) << "pad_left_right (pad_w) < 0";
	LOG_IF(FATAL, param->u() <= 0) << "vertical_filter_stride (u) <= 0";
	LOG_IF(FATAL, param->v() <= 0) << "horizontal_filter_stride (v) <= 0";
	LOG_IF(FATAL, param->dilation_h() <= 0) << "filter_height_dilation (dilation_h) <= 0";
	LOG_IF(FATAL, param->dilation_w() <= 0) << "filter_width_dilation (dilation_w) <= 0";
	DF_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_convDesc, param->pad_h(), param->pad_w(), param->u(), param->v(), param->dilation_h(), param->dilation_w(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));	
	int n, c, h, w;
	DF_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(_convDesc, _inputs[0]->value()->descriptor(), _filterDesc, &n, &c, &h, &w));	
	_outputs[0]->initValue({ n, c, h, w });
	LOG(INFO) << "Initializing Convolution " << _name << " - " << _inputs[0]->value()->shape() << " * " << _inputs[1]->value()->shape() << " -> " << _outputs[0]->value()->shape();	
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _inputs[0]->value()->descriptor(), _filterDesc, _convDesc, _outputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _inputs[0]->value()->descriptor(), _filterDesc, _convDesc, _outputs[0]->value()->descriptor(), _fwdAlgo, &_fwdWorkspaceSize));	
	_maxWorkspaceSize = _fwdWorkspaceSize; 

}

void Convolution2D::initBackward() {	
	_outputs[0]->initDiff();
	
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _filterDesc, _outputs[0]->diff()->descriptor(), _convDesc, _inputs[0]->diff()->descriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _filterDesc, _outputs[0]->diff()->descriptor(), _convDesc, _inputs[0]->diff()->descriptor(), _bwdDataAlgo, &_bwdDataWorkspaceSize));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(_cudnnHandle, _inputs[0]->value()->descriptor(), _outputs[0]->diff()->descriptor(), _convDesc, _filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_bwdFilterAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(_cudnnHandle, _inputs[0]->value()->descriptor(), _outputs[0]->diff()->descriptor(), _convDesc, _filterDesc, _bwdFilterAlgo, &_bwdFilterWorkspaceSize));
	_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdDataWorkspaceSize, _bwdFilterWorkspaceSize });
}

void Convolution2D::forward() {
	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		DF_CUDA_CHECK(cudaMalloc(&d_workspace, _maxWorkspaceSize));
	DF_CUDNN_CHECK(cudnnConvolutionForward(_cudnnHandle, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _filterDesc, _inputs[1]->value()->data(), _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()));	
}

void Convolution2D::backward() {	
	DF_CUDNN_CHECK(cudnnConvolutionBackwardFilter(_cudnnHandle, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _convDesc, _bwdFilterAlgo, d_workspace, _bwdFilterWorkspaceSize, &beta, _filterDesc, _inputs[1]->diff()->mutableData()));
	DF_CUDNN_CHECK(cudnnConvolutionBackwardData(_cudnnHandle, &alpha, _filterDesc, _inputs[1]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _convDesc, _bwdDataAlgo, d_workspace, _bwdDataWorkspaceSize, &beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()));
}
