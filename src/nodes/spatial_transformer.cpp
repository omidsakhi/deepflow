#include "nodes/spatial_transformer.h"

SpatialTransformer::SpatialTransformer(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_spatial_transformer_param() == false) << "param.has_spatial_transformer_param() == false";
}

void SpatialTransformer::init()
{	
	auto inputDims = _inputs[0]->dims();
	auto thetaDims = _inputs[1]->dims();
	auto gridDims = _inputs[2]->dims();
	int outputDims[4] = { inputDims[0], inputDims[1], gridDims[1], gridDims[2] };
	LOG_IF(FATAL, thetaDims[0] == 0 || thetaDims[1] != 1 || thetaDims[2] != 2 || thetaDims[3] != 3) << "[FAILED] " << _name << " - theta (second input) dimensions must be Nx1x2x3 but it was " << _inputs[1]->value()->shape();	
	LOG_IF(FATAL, inputDims[0] != thetaDims[0]) << "[FAILED] " << _name << " - Number of input samples (N) must match for input and theta";
	LOG_IF(FATAL, gridDims[0] != inputDims[0] || gridDims[3] != 2) << "[FAILED] " << _name << " - Grid (third input) dimension must be NxHxWx2";
	DF_NODE_CUDNN_CHECK(
		cudnnCreate(&_cudnnHandle)
	);
	DF_NODE_CUDNN_CHECK(
		cudnnCreateSpatialTransformerDescriptor(&_stDesc)
	);
	DF_NODE_CUDNN_CHECK(
		cudnnSetSpatialTransformerNdDescriptor(_stDesc, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, outputDims)
	);	
	LOG_IF(FATAL, _inputs[2]->diff() == nullptr) << "[FAILED] " << _name << " - Grid must have a place to store diff memory.";
	_outputs[0]->initValue({ outputDims[0], outputDims[1], outputDims[2], outputDims[3]});
	_outputs[0]->initDiff();
}

void SpatialTransformer::forward()
{
	DF_NODE_CUDNN_CHECK(
		cudnnSpatialTfGridGeneratorForward(_cudnnHandle, _stDesc, _inputs[1]->value()->gpu_data(), _inputs[2]->value()->gpu_data())
	);
	DF_NODE_CUDNN_CHECK(
		cudnnSpatialTfSamplerForward(_cudnnHandle, _stDesc, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(), _inputs[2]->value()->gpu_data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->gpu_data())
	);
}

void SpatialTransformer::backward()
{	
	if (_inputs[0]->diff()) {		
		DF_NODE_CUDNN_CHECK(
			cudnnSpatialTfSamplerBackward(_cudnnHandle, _stDesc,
				&one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->gpu_data(),
				&zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->gpu_data(),
				&one, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->gpu_data(),
				_inputs[2]->value()->gpu_data(), &zero, _inputs[2]->diff()->gpu_data())
		);
	}
	if (_inputs[1]->diff()) {
		DF_NODE_CUDNN_CHECK(
			cudnnSpatialTfGridGeneratorBackward(_cudnnHandle, _stDesc, _inputs[2]->diff()->gpu_data(), _inputs[1]->diff()->gpu_data())
		);
	}
}

std::string SpatialTransformer::to_cpp() const
{
	return std::string();
}
