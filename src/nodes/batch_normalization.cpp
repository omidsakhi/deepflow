#include "nodes/batch_normalization.h"

BatchNormalization::BatchNormalization(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_batch_normalization_param() == false) << "param.has_batch_normalization_param() == false";
}

void BatchNormalization::initForward()
{

	auto param = _param->batch_normalization_param();
	_exp_avg_factor = param.exp_avg_factor();
	auto input = _inputs[0];
	auto output = _outputs[0];
	auto inputDims = input->value()->dims();
	output->initValue(inputDims);
	_xDesc = input->value()->descriptor();
	_yDesc = output->value()->descriptor();
	_x = (float*)input->value()->mutableData();
	_y = (float*)output->value()->mutableData();
	DF_NODE_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_bnScaleBiasMeanVarDesc));
	if (param.mode() == deepflow::BatchNormalizationParam_Mode_CUDNN_BATCHNORM_PER_ACTIVATION) {
		_bnMode = CUDNN_BATCHNORM_PER_ACTIVATION;
		DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputDims[1], inputDims[2], inputDims[3]));
		_bnScaleBiasMeanVarSize = inputDims[1] * inputDims[2] * inputDims[3];
	}
	else if (param.mode() == deepflow::BatchNormalizationParam_Mode_CUDNN_BATCHNORM_SPATIAL) {
		_bnMode = CUDNN_BATCHNORM_SPATIAL;
		DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputDims[1], 1, 1));
		_bnScaleBiasMeanVarSize = inputDims[1];
	}
	else {
		LOG(FATAL) << "Unsupported batch normalization mode.";
	}
	LOG(INFO) << "Initializing Batch Normalization " << _name << " - " << _outputs[0]->value()->shape();
	_eps = CUDNN_BN_MIN_EPSILON;
	_bnScaleBiasMeanVarSizeInBytes = _bnScaleBiasMeanVarSize * sizeof(float);
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	
	DF_NODE_CUDA_CHECK(cudaMalloc(&_bnScale, _bnScaleBiasMeanVarSizeInBytes));
	if (param.has_scale()) {
		LOG_IF(FATAL, param.scale().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_bnScale, param.scale().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	}
	else {
		fill(_bnScaleBiasMeanVarSize, 1, _bnScale);
	}

	DF_NODE_CUDA_CHECK(cudaMalloc(&_bnBias, _bnScaleBiasMeanVarSizeInBytes));
	if (param.has_bias()) {
		LOG_IF(FATAL, param.bias().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_bnBias, param.bias().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	} else {		
		fill(_bnScaleBiasMeanVarSize, 0, _bnBias);
	}

	DF_NODE_CUDA_CHECK(cudaMalloc(&_bnMean, _bnScaleBiasMeanVarSizeInBytes));
	if (param.has_mean()) {
		LOG_IF(FATAL, param.mean().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_bnMean, param.mean().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	}
	else {
		fill(_bnScaleBiasMeanVarSize, 0, _bnMean);
	}

	DF_NODE_CUDA_CHECK(cudaMalloc(&_bnInvVariance, _bnScaleBiasMeanVarSizeInBytes));
	if (param.has_var()) {
		LOG_IF(FATAL, param.var().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_bnInvVariance, param.var().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	}
	else {		
		fill(_bnScaleBiasMeanVarSize, 0, _bnInvVariance);
	}

	if (_exp_avg_factor != 0) {		
		DF_NODE_CUDA_CHECK(cudaMalloc(&_resultRunningMean, _bnScaleBiasMeanVarSizeInBytes));
		fill(_bnScaleBiasMeanVarSize, 0, _resultRunningMean);
		DF_NODE_CUDA_CHECK(cudaMalloc(&_resultRunningVariance, _bnScaleBiasMeanVarSizeInBytes));
		fill(_bnScaleBiasMeanVarSize, 0, _resultRunningVariance);
	}
}

void BatchNormalization::initBackward()
{	
	_outputs[0]->initDiff();
	_dy = (float*)_outputs[0]->diff()->mutableData();
	_dx = (float*)_inputs[0]->diff()->mutableData();
}

void BatchNormalization::forward()
{
	if ((_context && (_context->phase_behaviour == deepflow::PhaseParam_PhaseBehaviour_TRAIN || _context->phase_behaviour == deepflow::PhaseParam_PhaseBehaviour_TRAIN_AND_INFERENCE)) || _context == nullptr) {		
		DF_NODE_CUDNN_CHECK(
			cudnnBatchNormalizationForwardTraining(
				_cudnnHandle,
				_bnMode,
				&one,
				&zero,
				_xDesc,
				_x,
				_yDesc,
				_y,
				_bnScaleBiasMeanVarDesc,
				_bnScale,
				_bnBias,
				_exp_avg_factor,
				_resultRunningMean,
				_resultRunningVariance,
				_eps,
				_bnMean,
				_bnInvVariance
			));		
	}
	else {		
		DF_NODE_CUDNN_CHECK(
			cudnnBatchNormalizationForwardInference(
				_cudnnHandle,
				_bnMode,
				&one,
				&zero,
				_xDesc,
				_x,
				_yDesc,
				_y,
				_bnScaleBiasMeanVarDesc,
				_bnScale,
				_bnBias,
				_bnMean,
				_bnInvVariance,
				_eps
			));
	}
}

void BatchNormalization::backward()
{	
	auto param = _param->batch_normalization_param();
	float ad = param.alpha_data();
	float bd = param.beta_data();
	float ap = param.alpha_param();
	float bp = param.beta_param();
	if (_inputs[0]->connectedNode()->propagateBack()) {
		DF_NODE_CUDNN_CHECK(
			cudnnBatchNormalizationBackward(
				_cudnnHandle,
				_bnMode,
				&ad,
				&bd,
				&ap,
				&bp,
				_xDesc,
				_x,
				_yDesc,
				_dy,
				_xDesc,
				_dx,
				_bnScaleBiasMeanVarDesc,
				_bnScale,
				_bnScale,
				_bnBias,
				_eps,
				_bnMean,
				_bnInvVariance));
	}
}

void BatchNormalization::prep_for_saving()
{	
	auto bn_param = _param->mutable_batch_normalization_param();

	auto mutable_scale = bn_param->mutable_scale();
	auto mutable_scale_data = mutable_scale->mutable_data();
	mutable_scale_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_scale_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_scale_data->mutable_data(), _bnScale, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));
	
	auto mutable_bias = bn_param->mutable_bias();
	auto mutable_bias_data = mutable_bias->mutable_data();
	mutable_bias_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_bias_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_bias_data->mutable_data(), _bnBias, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));

	auto mutable_mean = bn_param->mutable_mean();
	auto mutable_mean_data = mutable_mean->mutable_data();
	mutable_mean_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_mean_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_mean_data->mutable_data(), _bnMean, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));

	auto mutable_var = bn_param->mutable_var();
	auto mutable_var_data = mutable_var->mutable_data();
	mutable_var_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_var_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_var_data->mutable_data(), _bnInvVariance, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));

}

std::string BatchNormalization::to_cpp() const
{
	return std::string();
}
