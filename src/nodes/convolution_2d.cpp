#include "nodes/convolution_2d.h"

Convolution2D::Convolution2D(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_conv_2d_param() == false) << "param.has_conv_2d_param() [FAILED]";
	_num_inputs = param.input_size(); // COULD BE 2 WITHOUT BIAS OR 3 WITH BIAS
	d_workspace = 0;
}

int Convolution2D::minNumInputs()
{
	return _num_inputs;
}

void Convolution2D::initForward() {
	LOG_IF(FATAL, _num_inputs != 2 && _num_inputs != 3) << "_num_inputs != 2 && _num_inputs != 3";
	_xDesc = _inputs[0]->value()->descriptor();
	_x = _inputs[0]->value()->mutableData();
	auto inputDims = _inputs[0]->dims();	
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	DF_CUDNN_CHECK(cudnnCreateFilterDescriptor(&_wDesc));
	_w = _inputs[1]->value()->mutableData();
	auto filterDims = _inputs[1]->dims();	
	LOG_IF(FATAL, filterDims[1] != inputDims[1]) << "Input channels " << inputDims[1] << " != Filter channels " << filterDims[1];	
	DF_CUDNN_CHECK(cudnnSetFilter4dDescriptor(_wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3]));	
	deepflow::Conv2dParam *param = _param.mutable_conv_2d_param();
	DF_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&_convDesc));	
	LOG_IF(FATAL, param->pad_h() < 0) << "pad_top_bottom (pad_h) < 0";
	LOG_IF(FATAL, param->pad_w() < 0) << "pad_left_right (pad_w) < 0";
	LOG_IF(FATAL, param->u() <= 0) << "vertical_filter_stride (u) <= 0";
	LOG_IF(FATAL, param->v() <= 0) << "horizontal_filter_stride (v) <= 0";
	LOG_IF(FATAL, param->dilation_h() <= 0) << "filter_height_dilation (dilation_h) <= 0";
	LOG_IF(FATAL, param->dilation_w() <= 0) << "filter_width_dilation (dilation_w) <= 0";
	DF_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(_convDesc, param->pad_h(), param->pad_w(), param->u(), param->v(), param->dilation_h(), param->dilation_w(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));	
	int n, c, h, w;
	DF_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(_convDesc, _xDesc, _wDesc, &n, &c, &h, &w));
	_outputs[0]->initValue({ n, c, h, w });
	_yDesc = _outputs[0]->value()->descriptor();
	_y = _outputs[0]->value()->mutableData();	
	LOG(INFO) << "Initializing Convolution " << _name << " - " << _inputs[0]->value()->shape() << " * " << _inputs[1]->value()->shape() << " -> " << _outputs[0]->value()->shape();	
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(_cudnnHandle, _xDesc, _wDesc, _convDesc, _yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_fwdAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(_cudnnHandle, _xDesc, _wDesc, _convDesc, _yDesc, _fwdAlgo, &_fwdWorkspaceSize));	
	_maxWorkspaceSize = _fwdWorkspaceSize;

	// BIAS & ACTIVATION
	if (_num_inputs == 3) {
		_zDesc = _yDesc;				
		_z = _y;
		//DF_CUDA_CHECK(cudaMalloc(&_z, _outputs[0]->value()->sizeInBytes()));
		//DF_CUDA_CHECK(cudaMemset(_z, 0, _outputs[0]->value()->sizeInBytes()));
		_bDesc = _inputs[2]->value()->descriptor();
		_b = _inputs[2]->value()->mutableData();
		auto biasDims = _inputs[2]->dims();
		//if (biasDims[0] != 1 || biasDims[1] != filterDims[0] || biasDims[2] != h || biasDims[3] != w)
		//	LOG(FATAL) << "Dimension of the bias must be 1x" << filterDims[0] << "x" << h << "x" << w;
		//LOG_IF(FATAL, _inputs[2]->dims()[1] != _inputs[1]->dims()[0]) << "The second dimension of biasDesc and the first dimension of filterDesc are not equal.";
		DF_CUDNN_CHECK(cudnnCreateActivationDescriptor(&_activationDesc));
		cudnnSetActivationDescriptor(_activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1.0f);
	}
}

void Convolution2D::initBackward() {
	if (_inputs[0]->diff()) {
		_dxDesc = _inputs[0]->diff()->descriptor();
		_dx = _inputs[0]->diff()->mutableData();
	}
	_dw = _inputs[1]->diff()->mutableData();

	_outputs[0]->initDiff();
	_dyDesc = _outputs[0]->diff()->descriptor();
	_dy = _outputs[0]->diff()->mutableData();	
	
	if (_inputs[0]->diff()) {		
		DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(_cudnnHandle, _wDesc, _dyDesc, _convDesc, _dxDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &_bwdDataAlgo));
		DF_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnnHandle, _wDesc, _dyDesc, _convDesc, _dxDesc, _bwdDataAlgo, &_bwdDataWorkspaceSize));
		_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdDataWorkspaceSize});
	}
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(_cudnnHandle, _xDesc, _dyDesc, _convDesc, _wDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &_bwdFilterAlgo));
	DF_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(_cudnnHandle, _xDesc, _dyDesc, _convDesc, _wDesc, _bwdFilterAlgo, &_bwdFilterWorkspaceSize));
	_maxWorkspaceSize = std::max({ _maxWorkspaceSize, _bwdFilterWorkspaceSize });

	if (_num_inputs == 3) {
		_dzDesc = _dyDesc;
		DF_CUDA_CHECK(cudaMalloc(&_dz, _outputs[0]->diff()->sizeInBytes()));
		_db = _inputs[2]->diff()->mutableData();
		_dbDesc = _inputs[2]->diff()->descriptor();		
	}
}

void Convolution2D::forward() {
	if (d_workspace == 0 && _maxWorkspaceSize != 0)
		DF_CUDA_CHECK(cudaMalloc(&d_workspace, _maxWorkspaceSize));
	if (_num_inputs == 2) {
		DF_CUDNN_CHECK(cudnnConvolutionForward(_cudnnHandle, &one, _xDesc, _x, _wDesc, _w, _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &zero, _yDesc, _y));
	}
	else {
		DF_CUDNN_CHECK(cudnnConvolutionBiasActivationForward(_cudnnHandle, &one, _xDesc, _x, _wDesc, _w, _convDesc, _fwdAlgo, d_workspace, _fwdWorkspaceSize, &zero, _zDesc, _z, _bDesc, _b, _activationDesc, _yDesc, _y));
	}
}

void Convolution2D::backward() {
	if (_num_inputs == 3) {
		if (_inputs[2]->connectedNode()->propagateBack())
			cudnnConvolutionBackwardBias(_cudnnHandle, &one, _dyDesc, _dy, &one, _dbDesc, _db);
		cudnnActivationBackward(_cudnnHandle, _activationDesc, &one, _yDesc, _y, _dyDesc, _dy, _xDesc, _x, &one, _dzDesc, _dz);
	}
	else {
		_dz = _dy;
	}
	if (_inputs[0]->connectedNode()->propagateBack())
		DF_CUDNN_CHECK(cudnnConvolutionBackwardData(_cudnnHandle, &one, _wDesc, _w, _dzDesc, _dz, _convDesc, _bwdDataAlgo, d_workspace, _bwdDataWorkspaceSize, &one, _dxDesc, _dx));
	if (_inputs[1]->connectedNode()->propagateBack())
		DF_CUDNN_CHECK(cudnnConvolutionBackwardFilter(_cudnnHandle, &one, _xDesc, _x, _dzDesc, _dz, _convDesc, _bwdFilterAlgo, d_workspace, _bwdFilterWorkspaceSize, &one, _wDesc, _dw));
}

std::string Convolution2D::to_cpp() const
{	
	const deepflow::Conv2dParam &param = _param.conv_2d_param();
	std::string cpp = "auto " + _name + " = df.conv2d(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	if (_num_inputs == 3)
		cpp += _input_name_for_cpp(2) + ", ";
	else
		cpp += "\"\", ";
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
