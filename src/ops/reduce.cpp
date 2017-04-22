#include "ops/reduce.h"

Reduce::Reduce(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_reduce_param() == false) << "param.has_reduce_param() == false [FAILED]";
}

void Reduce::initForward() {
	
	const ReduceParam &reduceParam = _param.reduce_param();
	int reduceDim = reduceParam.reduce_dim();
	LOG_IF(FATAL, reduceDim > 3) << "reduceDim > 3 [FAILED]";
	_reduceTensorOp = (cudnnReduceTensorOp_t) reduceParam.reduce_op();

	auto inputDims = _inputs[0]->value()->dims();
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
		
	auto outputDims = inputDims;
	outputDims[reduceDim] = 1;

	_outputs[0]->initValue(outputDims, Tensor::Float);
	_outputs[1]->initValue(outputDims, Tensor::Int32);	
	
	if (requiresIndices())
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
	else
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
	
	std::string opString;
	switch (reduceParam.reduce_op()) {
	case ReduceParam_ReduceOp_ADD:
		opString = "ReduceSum";
		break;
	case ReduceParam_ReduceOp_MUL:
		opString = "ReduceMul";
		break;
	case ReduceParam_ReduceOp_MIN:
		opString = "ReduceMin";
		break;
	case ReduceParam_ReduceOp_MAX:
		opString = "ReduceMax";
		break;
	case ReduceParam_ReduceOp_AMAX:
		opString = "ReduceMaxAbs";
		break;
	case ReduceParam_ReduceOp_AVG:
		opString = "ReduceMean";
		break;
	case ReduceParam_ReduceOp_NORM1:
		opString = "ReduceSumAbs";
		break;
	case ReduceParam_ReduceOp_NORM2:
		opString = "ReduceHypot";
		break;
	};

	DF_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_reduceTensorDesciptor));
	DF_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_reduceTensorDesciptor, _reduceTensorOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, _reduceTensorIndices, CUDNN_32BIT_INDICES));
	DF_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(_cudnnHandle, _reduceTensorDesciptor, _inputs[0]->value()->descriptor(), _outputs[0]->value()->descriptor(), &_workspaceSizeInBytes));
	DF_CUDA_CHECK(cudaMalloc(&_d_workspace, _workspaceSizeInBytes));


	LOG(INFO) << "Initializing " << opString << " " << _name << " - " << _inputs[0]->value()->shape() << " -> " << _outputs[0]->value()->shape();
}

bool Reduce::requiresIndices() {
	return (_reduceTensorOp == CUDNN_REDUCE_TENSOR_MAX || _reduceTensorOp == CUDNN_REDUCE_TENSOR_MIN);
}

void Reduce::initBackward() {
	
}

void Reduce::forward() {		
	DF_CUDNN_CHECK(
		cudnnReduceTensor(
			_cudnnHandle,
			_reduceTensorDesciptor,
			requiresIndices() ? _outputs[1]->value()->mutableData() : 0,
			requiresIndices() ? _outputs[1]->value()->sizeInBytes() : 0,
			_d_workspace,
			_workspaceSizeInBytes,
			&alpha,
			_inputs[0]->value()->descriptor(),
			_inputs[0]->value()->data(),
			&beta,
			_outputs[0]->value()->descriptor(),
			_outputs[0]->value()->mutableData()));
}

void Reduce::backward() {

}
