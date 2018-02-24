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
	_x = (float*)input->value()->mutableData();
	_y = (float*)output->value()->mutableData();
	
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
	
	if (cache) {
		DF_NODE_CUDA_CHECK(cudaMalloc(&_resultSaveMean, _bnScaleBiasMeanVarSizeInBytes));
		if (param.has_mean()) {
			LOG_IF(FATAL, param.mean().data_size() != _bnScaleBiasMeanVarSize);
			DF_NODE_CUDA_CHECK(cudaMemcpy(_resultSaveMean, param.mean().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));			
		}
		else {
			fill(_bnScaleBiasMeanVarSize, 0, _resultSaveMean);
		}		
		DF_NODE_CUDA_CHECK(cudaMalloc(&_resultSaveInvVariance, _bnScaleBiasMeanVarSizeInBytes));
		if (param.has_var()) {
			LOG_IF(FATAL, param.var().data_size() != _bnScaleBiasMeanVarSize);
			DF_NODE_CUDA_CHECK(cudaMemcpy(_resultSaveInvVariance, param.var().data().data(), _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyHostToDevice));
		}
		else {
			fill(_bnScaleBiasMeanVarSize, 0, _resultSaveInvVariance);
		}
	}	

	_bnScale = (float*)_inputs[1]->value()->data();
	_bnBias = (float*)_inputs[2]->value()->data();

	_outputs[0]->initDiff();
	_dy = (float*)_outputs[0]->diff()->mutableData();
	_dx = (float*)_inputs[0]->diff()->mutableData();
	_resultBnScaleDiff = (float*)_inputs[1]->diff()->mutableData();
	_resultBnBiasDiff = (float*)_inputs[2]->diff()->mutableData();
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
				0,
				nullptr,
				nullptr,
				_eps,
				_resultSaveMean,
				_resultSaveInvVariance
			));		
	}
	else {
		LOG(FATAL);
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
				_resultSaveMean,
				_resultSaveInvVariance,
				_eps
			));
	}
}

void BatchNormalization::backward()
{		
	if (_inputs[0]->connectedNode()) {
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
				_resultSaveMean,
				_resultSaveInvVariance));
	}	
}

void BatchNormalization::prep_for_saving()
{	
	
	auto bn_param = _param->mutable_batch_normalization_param();

	auto mutable_mean = bn_param->mutable_mean();
	auto mutable_mean_data = mutable_mean->mutable_data();
	mutable_mean_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_mean_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_mean_data->mutable_data(), _resultSaveMean, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));

	auto mutable_var = bn_param->mutable_var();
	auto mutable_var_data = mutable_var->mutable_data();
	mutable_var_data->Resize(_bnScaleBiasMeanVarSize, 0);
	LOG_IF(FATAL, mutable_var_data->size() != _bnScaleBiasMeanVarSize);
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_var_data->mutable_data(), _resultSaveInvVariance, _bnScaleBiasMeanVarSizeInBytes, cudaMemcpyDeviceToHost));
	
}

std::string BatchNormalization::to_cpp() const
{
	return std::string();
}
