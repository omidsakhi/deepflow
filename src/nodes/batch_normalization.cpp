#include "nodes/batch_normalization.h"

BatchNormalization::BatchNormalization(deepflow::NodeParam *param) : Node(param)
{
	LOG_IF(FATAL, param->has_batch_normalization_param() == false) << "param.has_batch_normalization_param() == false";
}

void BatchNormalization::init()
{

	auto param = _param->batch_normalization_param();
	bool cache = param.cache_meanvar();
	auto input = _inputs[0];
	auto output = _outputs[0];
	auto inputDims = input->value()->dims();
	output->initValue(inputDims);
	_xDesc = input->value()->descriptor();
	_yDesc = output->value()->descriptor();
	
	std::array<int, 4> scaleBiasExpectedDims;

	DF_NODE_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_bnScaleBiasMeanVarDesc));

	if (param.mode() == deepflow::BatchNormalizationParam_Mode_CUDNN_BATCHNORM_PER_ACTIVATION) {
		_bnMode = CUDNN_BATCHNORM_PER_ACTIVATION;
		DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputDims[1], inputDims[2], inputDims[3]));
		_bnScaleBiasMeanVarSize = inputDims[1] * inputDims[2] * inputDims[3];
		scaleBiasExpectedDims = { 1, inputDims[1], inputDims[2], inputDims[3] };
	}
	else if (param.mode() == deepflow::BatchNormalizationParam_Mode_CUDNN_BATCHNORM_SPATIAL) {
		_bnMode = CUDNN_BATCHNORM_SPATIAL;
		DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputDims[1], 1, 1));
		_bnScaleBiasMeanVarSize = inputDims[1];
		scaleBiasExpectedDims = { 1, inputDims[1], 1, 1 };
	}
	else {
		LOG(FATAL) << "Unsupported batch normalization mode.";
	}		
	_bnScaleBiasMeanVarSizeInBytes = _bnScaleBiasMeanVarSize * sizeof(float);
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	
	LOG_IF(FATAL, _inputs[1]->value()->dims() != scaleBiasExpectedDims || _inputs[2]->value()->dims() != scaleBiasExpectedDims) << _name << ": Expected dimention of scale and bias vector are " <<
		scaleBiasExpectedDims[0] << "x" << scaleBiasExpectedDims[1] << "x" << scaleBiasExpectedDims[2] << "x" << scaleBiasExpectedDims[3];

	DF_NODE_CUDA_CHECK(cudaMalloc(&_runningMean, _bnScaleBiasMeanVarSizeInBytes));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_runningVariance, _bnScaleBiasMeanVarSizeInBytes));
	if (param.has_mean()) {
		LOG_IF(FATAL, param.mean().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_runningMean, param.mean().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	}
	else {		
		fill(_bnScaleBiasMeanVarSize, 0, _runningMean);
	}

	if (param.has_var()) {
		LOG_IF(FATAL, param.var().data_size() != _bnScaleBiasMeanVarSize);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_runningVariance, param.var().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
	}
	else {
		cudnnSetTensor(_cudnnHandle, _bnScaleBiasMeanVarDesc, _runningVariance, (void*)&one);		
	}

	if (cache) {
		DF_NODE_CUDA_CHECK(cudaMalloc(&_cachedMean, _bnScaleBiasMeanVarSizeInBytes));
		DF_NODE_CUDA_CHECK(cudaMalloc(&_cachedVariance, _bnScaleBiasMeanVarSizeInBytes));
	}	

	_exp_avg_factor = param.exp_avg_factor();
	_eps = param.eps();
	if (_eps < CUDNN_BN_MIN_EPSILON)
		_eps = CUDNN_BN_MIN_EPSILON;
	LOG_IF(FATAL, _exp_avg_factor == 0) << "Average factor cannot be zero.";

	_outputs[0]->initDiff();		
}

void BatchNormalization::forward()
{
	float * _x = _inputs[0]->value()->gpu_data(DF_LINE);
	float * _y = _outputs[0]->value()->gpu_data(DF_LINE);
	float *_bnScale = _inputs[1]->value()->gpu_data(DF_LINE);
	float *_bnBias = _inputs[2]->value()->gpu_data(DF_LINE);
	if (_context->execution_mode == ExecutionContext::TRAIN) {
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
				_runningMean,
				_runningVariance,
				_eps,
				_cachedMean,
				_cachedVariance
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
				_runningMean,
				_runningVariance,
				_eps
			));
	}	
}

void BatchNormalization::backward()
{		
	if (_inputs[0]->diff()) {
		float *_dy = _outputs[0]->diff()->gpu_data(DF_LINE);
		float *_dx = _inputs[0]->diff()->gpu_data(DF_LINE);
		float * _x = _inputs[0]->value()->gpu_data(DF_LINE);
		float *_bnScale = _inputs[1]->value()->gpu_data(DF_LINE);		
		float * _resultBnScaleDiff = (float*)_inputs[1]->diff()->gpu_data(DF_LINE);
		float * _resultBnBiasDiff = (float*)_inputs[2]->diff()->gpu_data(DF_LINE);
		DF_NODE_CUDNN_CHECK(
			cudnnBatchNormalizationBackward(
				_cudnnHandle,
				_bnMode,
				&one,
				&zero,
				&one,
				&zero,
				_xDesc,
				_x,
				_yDesc,
				_dy,
				_xDesc,
				_dx,
				_bnScaleBiasMeanVarDesc,
				_bnScale,
				_resultBnScaleDiff,
				_resultBnBiasDiff,
				_eps,
				_cachedMean,
				_cachedVariance));
	}	
}

void BatchNormalization::prep_for_saving()
{	
	
	auto bn_param = _param->mutable_batch_normalization_param();

	auto mutable_mean = bn_param->mutable_mean();
	auto mutable_mean_data = mutable_mean->mutable_data();
	mutable_mean_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_mean_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_mean_data->mutable_data(), _runningMean, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));

	auto mutable_var = bn_param->mutable_var();
	auto mutable_var_data = mutable_var->mutable_data();
	mutable_var_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_var_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_var_data->mutable_data(), _runningVariance, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));
	
}

std::string BatchNormalization::to_cpp() const
{
	auto param = _param->batch_normalization_param();
	//std::string batch_normalization(std::string input, std::string scale, std::string bias, NormalizationMode mode = DeepFlow::SPATIAL, bool cache = true, std::string name = "batch_norm", std::initializer_list<std::string> phases = {});
	std::string cpp = "auto " + _name + " = df.batch_normalization(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += _input_name_for_cpp(2) + ", ";
	if (param.mode() == deepflow::BatchNormalizationParam_Mode_CUDNN_BATCHNORM_SPATIAL)
		cpp += "DeepFlow::SPATIAL ,";
	else 
		cpp += "DeepFlow::PER_ACTIVATION ,";
	cpp += param.cache_meanvar() ? " true, " : " false, ";
	cpp += "\"" + _name + "\");";	
	return cpp;	
}
