#include "nodes/loss.h"
#include "core/common_cu.h"

Loss::Loss(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_loss_param() == false) << "param.has_loss_param() == false";
}

void Loss::init() {
	auto lossParam = _param->loss_param();
	_reduceTensorOp = (cudnnReduceTensorOp_t)lossParam.reduce_op();
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));
	_reduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES;
	_outputs[0]->initValue({ 1,1,1,1 });
	_alpha = _param->loss_param().alpha();
	_beta = _param->loss_param().beta();	
	cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes());
	DF_NODE_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_reduceTensorDesciptor));
	DF_NODE_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_reduceTensorDesciptor, _reduceTensorOp, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, _reduceTensorIndices, CUDNN_32BIT_INDICES));
	DF_NODE_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(_cudnnHandle, _reduceTensorDesciptor, _inputs[0]->value()->descriptor(), _outputs[0]->value()->descriptor(), &_workspaceSizeInBytes));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_workspace, _workspaceSizeInBytes));	
}

void Loss::forward() {
	if (_outputs[0]->connectedNodes().size() > 0) {
		if (_inputs[0]->value()->size() == 1) {
			cpy(1, 1, _inputs[0]->value()->data(), 0, _outputs[0]->value()->mutableData());
		}
		else {
			DF_NODE_CUDNN_CHECK(
				cudnnReduceTensor(
					_cudnnHandle,
					_reduceTensorDesciptor,
					nullptr,
					0,
					_d_workspace,
					_workspaceSizeInBytes,
					&one,
					_inputs[0]->value()->descriptor(),
					_inputs[0]->value()->data(),
					&zero,
					_outputs[0]->value()->descriptor(),
					_outputs[0]->value()->mutableData())
			);
		}
	}	
}

void Loss::backward() {	
	cpy(_inputs[0]->value()->size(), _alpha, _inputs[0]->value()->data(), _beta, _inputs[0]->diff()->mutableData());	
}

std::string Loss::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.loss(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}

void Loss::set_alpha(float value)
{
	_alpha = value;
}

void Loss::set_beta(float value)
{
	_beta = value;
}
