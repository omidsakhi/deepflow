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
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0) << "cudnnCreate [FAILED]";	
		
	auto outputDims = inputDims;
	outputDims[reduceDim] = 1;

	_outputs[0]->initValue(outputDims, Tensor::Float);
	_outputs[1]->initValue(outputDims, Tensor::Int32);	
	
	if (requiresIndices())
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
	else
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
	
	LOG_IF(FATAL, cudnnCreateReduceTensorDescriptor(&_reduceTensorDesciptor) != 0) << "cudnnCreateReduceTensorDescriptor [FAILED]";
	LOG_IF(FATAL, cudnnSetReduceTensorDescriptor(_reduceTensorDesciptor, _reduceTensorOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, _reduceTensorIndices, CUDNN_32BIT_INDICES) != 0) << "cudnnSetReduceTensorDescriptor [FAILED]";
	LOG_IF(FATAL, cudnnGetReductionWorkspaceSize(_cudnnHandle, _reduceTensorDesciptor, _inputs[0]->value()->descriptor(), _outputs[0]->value()->descriptor(), &_workspaceSizeInBytes) != 0) << "cudnnGetReductionWorkspaceSize [FAILED]" ;
	LOG_IF(FATAL, cudaMalloc(&_d_workspace, _workspaceSizeInBytes) != 0) << " cudaMalloc(&_d_workspace, ... [FAILED]";
}

bool Reduce::requiresIndices() {
	return (_reduceTensorOp == CUDNN_REDUCE_TENSOR_MAX || _reduceTensorOp == CUDNN_REDUCE_TENSOR_MIN);
}

void Reduce::initBackward() {
	
}

void Reduce::forward() {		
	cudnnStatus_t t =
		cudnnReduceTensor(
			_cudnnHandle,
			_reduceTensorDesciptor,
			requiresIndices()?_outputs[1]->value()->mutableData():0,
			requiresIndices()?_outputs[1]->value()->sizeInBytes():0,
			_d_workspace,
			_workspaceSizeInBytes,
			&alpha,
			_inputs[0]->value()->descriptor(),
			_inputs[0]->value()->data(),
			&beta,
			_outputs[0]->value()->descriptor(),
			_outputs[0]->value()->mutableData());
	LOG_IF(FATAL,t != 0) << "cudnnReduceTensor - Op: " << ReduceParam_ReduceOp_Name((ReduceParam_ReduceOp) _reduceTensorOp) << " - " << cudaStatusToString(t) << " [FAILED]";
}

void Reduce::backward() {

}
