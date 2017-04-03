#include "ops/reduce.h"

Reduce::Reduce(NodeParam param) : Node(param) {
	LOG_IF(FATAL, param.has_op_reduce_param() == false) << "param.has_op_reduce_param() == false [FAILED]";
}

void Reduce::initForward() {
	
	const OpReduceParam &reduceParam = _param.op_reduce_param();
	int reduceDim = reduceParam.reduce_dim();
	LOG_IF(FATAL, reduceDim > 3) << "reduceDim > 3 [FAILED]";
	_reduceTensorOp = (cudnnReduceTensorOp_t) reduceParam.reduce_op();

	auto inputDims = _inputs[0]->value()->dims();
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0) << "cudnnCreate [FAILED]";	
		
	auto outputDims = inputDims;
	outputDims[reduceDim] = 1;

	_outputs[0]->initValue(outputDims, Tensor::Float);
	_outputs[1]->initValue(outputDims, Tensor::Int32);	
	
	if (_reduceTensorOp == CUDNN_REDUCE_TENSOR_MAX || _reduceTensorOp == CUDNN_REDUCE_TENSOR_MIN)
			_reduceTensotIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
	else
		_reduceTensotIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;

	LOG_IF(FATAL, cudnnCreateReduceTensorDescriptor(&_reduceTensorDesciptor) != 0) << "cudnnCreateReduceTensorDescriptor [FAILED]";
	LOG_IF(FATAL, cudnnSetReduceTensorDescriptor(_reduceTensorDesciptor, _reduceTensorOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, _reduceTensotIndices, CUDNN_32BIT_INDICES) != 0) << "cudnnSetReduceTensorDescriptor [FAILED]";
	LOG_IF(FATAL, cudnnGetReductionWorkspaceSize(_cudnnHandle, _reduceTensorDesciptor, _inputs[0]->value()->descriptor(), _outputs[0]->value()->descriptor(), &_workspaceSizeInBytes) != 0) << "cudnnGetReductionWorkspaceSize [FAILED]" ;
	LOG_IF(FATAL, cudaMalloc(&_d_workspace, _workspaceSizeInBytes) != 0) << " cudaMalloc(&_d_workspace, ... [FAILED]";
}

void Reduce::initBackward() {
	
}

void Reduce::forward() {		
	LOG_IF(FATAL,
		cudnnReduceTensor(
			_cudnnHandle,
			_reduceTensorDesciptor,
			_outputs[1]->value()->mutableData(),
			_outputs[1]->value()->sizeInBytes(),
			_d_workspace,
			_workspaceSizeInBytes,
			&alpha,
			_inputs[0]->value()->descriptor(),
			_inputs[0]->value()->data(),
			&beta,
			_outputs[0]->value()->descriptor(),
			_outputs[0]->value()->mutableData())
	!= 0) << "cudnnReduceTensor [FAILED]";
}

void Reduce::backward() {

}
