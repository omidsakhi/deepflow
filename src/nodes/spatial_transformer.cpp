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
	LOG_IF(FATAL, thetaDims[0] == 0 || thetaDims[1] != 1 || thetaDims[2] != 2 || thetaDims[3] != 3) << "[FAILED] " << _name << " - theta (second input) dimensions must be Nx1x2x3 but it was " << _inputs[1]->value()->shape();	
	LOG_IF(FATAL, inputDims[0] != thetaDims[0]) << "[FAILED] " << _name << " - Number of input samples (N) must match for input and theta";
	LOG_IF(FATAL, gridDims[0] != inputDims[0] || gridDims[1] != inputDims[2] || gridDims[2] != inputDims[3] || gridDims[3] != 2) << "[FAILED] " << _name << " - Grid (third input) dimension must be NxHxWx2";
	DF_NODE_CUDNN_CHECK(
		cudnnCreate(&_cudnnHandle)
	);
	DF_NODE_CUDNN_CHECK(
		cudnnCreateSpatialTransformerDescriptor(&_stDesc)
	);
	int nbDims = 4;
	int dimA[4] = { inputDims[0], inputDims[1], inputDims[2], inputDims[3] };
	DF_NODE_CUDNN_CHECK(
		cudnnSetSpatialTransformerNdDescriptor(_stDesc, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, nbDims, dimA)
	);	
	LOG_IF(FATAL, _inputs[2]->diff() == nullptr) << "[FAILED] " << _name << " - Grid must have a place to store diff memory.";
	_outputs[0]->initValue(inputDims);
	_outputs[0]->initDiff();
}

void SpatialTransformer::forward()
{
	DF_NODE_CUDNN_CHECK(
		cudnnSpatialTfGridGeneratorForward(_cudnnHandle, _stDesc, _inputs[1]->value()->data(), _inputs[2]->value()->mutableData())
	);
	DF_NODE_CUDNN_CHECK(
		cudnnSpatialTfSamplerForward(_cudnnHandle, _stDesc, &one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), _inputs[2]->value()->data(), &zero, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData())
	);
}

void SpatialTransformer::backward()
{	
	if (_inputs[0]->diff()) {		
		DF_NODE_CUDNN_CHECK(
			cudnnSpatialTfSamplerBackward(_cudnnHandle, _stDesc,
				&one, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(),
				&zero, _inputs[0]->diff()->descriptor(), _inputs[0]->diff()->mutableData(),
				&one, _outputs[0]->diff()->descriptor(), _outputs[0]->diff()->data(),
				_inputs[2]->value()->data(), &zero, _inputs[2]->diff()->mutableData())
		);
	}
	if (_inputs[1]->diff()) {
		DF_NODE_CUDNN_CHECK(
			cudnnSpatialTfGridGeneratorBackward(_cudnnHandle, _stDesc, _inputs[2]->diff()->data(), _inputs[1]->diff()->mutableData())
		);
	}
}

std::string SpatialTransformer::to_cpp() const
{
	return std::string();
}
