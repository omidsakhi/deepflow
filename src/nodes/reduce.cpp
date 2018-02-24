#include "nodes/reduce.h"

Reduce::Reduce(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_reduce_param() == false) << "param.has_reduce_param() == false [FAILED]";
}

std::string Reduce::op_name() const
{
	std::string op;
	switch (_reduceTensorOp) {
	case deepflow::ReduceParam_ReduceOp_ADD:
		op = "reduce_sum";
		break;
	case deepflow::ReduceParam_ReduceOp_MUL:
		op = "reduce_mul";
		break;
	case deepflow::ReduceParam_ReduceOp_MIN:
		if (_type == deepflow::ReduceParam_OutputType_VALUES)
			op = "reduce_min";
		else
			op = "argmin";
		break;
	case deepflow::ReduceParam_ReduceOp_MAX:
		if (_type == deepflow::ReduceParam_OutputType_VALUES)
			op = "reduce_max";
		else
			op = "argmax";
		break;
	case deepflow::ReduceParam_ReduceOp_AMAX:
		op = "reduce_absmax";
		break;
	case deepflow::ReduceParam_ReduceOp_AVG:
		op = "reduce_mean";
		break;
	case deepflow::ReduceParam_ReduceOp_NORM1:
		op = "reduce_norm1";
		break;
	case deepflow::ReduceParam_ReduceOp_NORM2:
		op = "reduce_norm2";
		break;
	};

	return op;
}

void Reduce::init() {
	
	auto reduceParam = _param->reduce_param();
	int reduceDim = reduceParam.reduce_dim();
	LOG_IF(FATAL, reduceDim > 3) << "reduceDim > 3 [FAILED]";
	_reduceTensorOp = (cudnnReduceTensorOp_t) reduceParam.reduce_op();
	_type = reduceParam.output_type();

	auto inputDims = _inputs[0]->value()->dims();
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
		
	auto outputDims = inputDims;
	outputDims[reduceDim] = 1;

	_outputs[0]->initValue(outputDims, Tensor::Float);
	_outputs[1]->initValue(outputDims, Tensor::Int32);	
	
	if (requiresIndices())
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
	else
		_reduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
	
	DF_NODE_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_reduceTensorDesciptor));	
	DF_NODE_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_reduceTensorDesciptor, _reduceTensorOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, _reduceTensorIndices, CUDNN_32BIT_INDICES));
	DF_NODE_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(_cudnnHandle, _reduceTensorDesciptor, _inputs[0]->value()->descriptor(), _outputs[0]->value()->descriptor(), &_workspaceSizeInBytes));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_workspace, _workspaceSizeInBytes));	
}

bool Reduce::requiresIndices() {
	return (_reduceTensorOp == CUDNN_REDUCE_TENSOR_MAX || _reduceTensorOp == CUDNN_REDUCE_TENSOR_MIN);
}

std::string Reduce::to_cpp() const
{
	auto reduceParam = _param->reduce_param();
	int reduceDim = reduceParam.reduce_dim();

	std::string cpp = "auto " + _name + " = df." + op_name() + "(" + _input_name_for_cpp(0) + ", " + std::to_string(reduceDim) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}

void Reduce::forward() {		
	DF_NODE_CUDNN_CHECK(
		cudnnReduceTensor(
			_cudnnHandle,
			_reduceTensorDesciptor,
			requiresIndices() ? _outputs[1]->value()->mutableData() : 0,
			requiresIndices() ? _outputs[1]->value()->sizeInBytes() : 0,
			_d_workspace,
			_workspaceSizeInBytes,
			&one,
			_inputs[0]->value()->descriptor(),
			_inputs[0]->value()->data(),
			&zero,
			_outputs[0]->value()->descriptor(),
			_outputs[0]->value()->mutableData()));
}

void Reduce::backward() {

}
