#include "ops/convolution_2d.h"

Convolution2D::Convolution2D(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_conv_2d_param() == false) << "param.has_op_conv_2d_param() [FAILED]";
	d_workspace = 0;
}

void Convolution2D::initForward() {	
	auto inputDims = _inputs[0]->dims();	
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0);	
	LOG_IF(FATAL, cudnnCreateFilterDescriptor(&_filterDesc) != 0);
	auto filterDims = _inputs[1]->dims();
	auto filterValueTensor = _inputs[1]->value();
	LOG_IF(FATAL, filterDims[1] != inputDims[1]) << "Input channels " << inputDims[1] << " != Filter channels " << filterDims[1];
	LOG_IF(FATAL, cudnnSetFilter4dDescriptor(_filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3]) != 0);
	Conv2dParam *param = _param.mutable_conv_2d_param();
	LOG_IF(FATAL, cudnnCreateConvolutionDescriptor(&_convDesc) != 0);
	LOG_IF(FATAL, param->pad_h() < 0) << "pad_top_bottom (pad_h) < 0";
	LOG_IF(FATAL, param->pad_w() < 0) << "pad_left_right (pad_w) < 0";
	LOG_IF(FATAL, param->u() <= 0) << "vertical_filter_stride (u) <= 0";
	LOG_IF(FATAL, param->v() <= 0) << "horizontal_filter_stride (v) <= 0";
	LOG_IF(FATAL, param->dilation_h() <= 0) << "filter_height_dilation (dilation_h) <= 0";
	LOG_IF(FATAL, param->dilation_w() <= 0) << "filter_width_dilation (dilation_w) <= 0";
	LOG_IF(FATAL, cudnnSetConvolution2dDescriptor(_convDesc, param->pad_h(), param->pad_w(), param->u(), param->v(), param->dilation_h(), param->dilation_w(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != 0);
	int n, c, h, w;
	LOG_IF(FATAL, cudnnGetConvolution2dForwardOutputDim(_convDesc, _inputs[0]->value()->descriptor(), _filterDesc, &n, &c, &h, &w) != 0);	
	_outputs[0]->initValue({ n, c, h, w });
	LOG(INFO) << "Initializing Convolution " << _name << " - " << _inputs[0]->value()->shape() << " * " << _inputs[1]->value()->shape() << " -> " << _outputs[0]->value()->shape();	
	LOG_IF(FATAL, cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _inputs[0]->value()->descriptor(), _filterDesc, _convDesc, _outputs[0]->value()->descriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo) != 0);
	LOG_IF(FATAL, cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _inputs[0]->value()->descriptor(), _filterDesc, _convDesc, _outputs[0]->value()->descriptor(), _fwdAlgo, &_fwdWorkspaceSize) != 0);
	_maxWorkspaceSize = _fwdWorkspaceSize; 

}

void Convolution2D::initBackward() {	
	_outputs[0]->initDiff();
	LOG_IF(FATAL, cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _filterDesc, _outputs[0]->diff()->descriptor(), _convDesc, _inputs[0]->diff()->descriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo) != 0) << "cudnnGetConvolutionBackwardDataAlgorithm [FAILED]";
	LOG_IF(FATAL, cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _filterDesc, _outputs[0]->diff()->descriptor(), _convDesc, _inputs[0]->diff()->descriptor(), _bwdDataAlgo, &_bwdDataWorkspaceSize) != 0) << "cudnnGetConvolutionBackwardDataWorkspaceSize [FAILED]";
	LOG_IF(FATAL, cudnnGetConvolutionBackwardFilterAlgorithm(_cudnnHandle, _inputs[0]->value()->descriptor(), _outputs[0]->diff()->descriptor(), _convDesc, _filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_bwdFilterAlgo) != 0) << "cudnnGetConvolutionBackwardFilterAlgorithm [FAILED]";
	LOG_IF(FATAL, cudnnGetConvolutionBackwardFilterWorkspaceSize(_cudnnHandle, _inputs[0]->value()->descriptor(), _outputs[0]->diff()->descriptor(), _convDesc, _filterDesc, _bwdFilterAlgo, &_bwdFilterWorkspaceSize) != 0) << "cudnnGetConvolutionBackwardFilterWorkspaceSize [FAILED]";
	_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdDataWorkspaceSize, _bwdFilterWorkspaceSize });
}

void Convolution2D::forward() {
	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		LOG_IF(FATAL, cudaMalloc(&d_workspace, _maxWorkspaceSize) != 0) << "cudaMalloc(&d_workspace ... [FAILED]";
	LOG_IF(FATAL, cudnnConvolutionForward(_cudnnHandle, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _filterDesc, _inputs[1]->value()->data(), _convDesc, _fwdAlgo, d_workspace, _maxWorkspaceSize, &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()) != 0) << "cudnnConvolutionForward [FAILED]";
}

void Convolution2D::backward() {
	LOG_IF(FATAL, cudnnConvolutionBackwardData(_cudnnHandle, &alpha, _filterDesc, _inputs[1]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _convDesc, _bwdDataAlgo, d_workspace, _maxWorkspaceSize, &beta, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData()) != 0) << "cudnnConvolutionBackwardData [FAILED]";
	LOG_IF(FATAL,cudnnConvolutionBackwardFilter(_cudnnHandle, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(), _convDesc, _bwdFilterAlgo, d_workspace, _maxWorkspaceSize, &beta, _filterDesc, _inputs[1]->diff()->mutableData()) != 0) << "cudnnConvolutionBackwardFilter [FAILED]";
}
