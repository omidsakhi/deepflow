
#include "nodes/variable.h"
#include "core/initializer.h"

#include <string>
#include <iostream>

#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

__global__
void VariableClampKernel(const int n, float *x, const float min, const float max)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float val = x[i];
		if (val < min)
			x[i] = min;
		else if (val > max)
			x[i] = max;
	}
}

Variable::Variable(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_variable_param() == false) << "param.has_variable_param() == false";
	_initializer = initializer;			
}

void Variable::init() {

	_initializer->init();
	_outputs[0]->initValue(_initializer->dims());	
	if (_param->variable_param().has_weights()) {		
		auto weights = _param->variable_param().weights();
		LOG_IF(FATAL, weights.data_size() != _outputs[0]->value()->size()) << "weights.weight_size() != _outputs[0]->value()->size() in " << _name << " - " << weights.data_size() << " != " << _outputs[0]->value()->size();
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), weights.data().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else if (_initializer->param()->has_init_data()) {		
		auto weights = _initializer->param()->init_data();
		LOG_IF(FATAL, weights.data_size() != _outputs[0]->value()->size()) << "weights.weight_size() != _outputs[0]->value()->size() in " << _name << " - " << weights.data_size() << " != " << _outputs[0]->value()->size();
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), weights.data().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else {		
		_initializer->apply(this);
		for (int i = 0; i < _outputs[0]->value()->size(); ++i)
			_param->mutable_variable_param()->mutable_init_param()->mutable_init_data()->add_data(0);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_param->mutable_variable_param()->mutable_init_param()->mutable_init_data()->mutable_data()->mutable_data(),_outputs[0]->value()->data(),_outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToHost));
	}

	_outputs[0]->initDiff();
	int size = _outputs[0]->value()->sizeInBytes();
	DF_CUDA_CHECK(cudaMalloc(&_grad, size));
	DF_CUDA_CHECK(cudaMemset(_grad, 0, size));

}

inline void Variable::forward() {
	
}

inline void Variable::backward() {
	if (!_param->variable_param().solver_name().empty()) {
		LOG_IF(INFO, _verbose > 2) << _name << " + gradients";
		cpy(_outputs[0]->value()->size(), 1.0, _outputs[0]->diff()->data(), 1.0, _grad);
	}
}

float * Variable::gradients()
{	
	return _grad;
}

void Variable::reset_gradients()
{	
	LOG_IF(INFO, _verbose > 3) << _name << " : gradients <- 0";
	_outputs[0]->resetDiff();	
	DF_CUDA_CHECK(cudaMemset(_grad, 0, _outputs[0]->value()->sizeInBytes()));
}

void Variable::prep_for_saving()
{	
	auto var_param = _param->mutable_variable_param();
	auto mutable_weights = var_param->mutable_weights();
	auto mutable_weights_data = mutable_weights->mutable_data();
	mutable_weights_data->Resize(_outputs[0]->value()->size(),0.0f);
	LOG_IF(FATAL, mutable_weights_data->size() != _outputs[0]->value()->size());
	DF_NODE_CUDA_CHECK(cudaMemcpy(mutable_weights_data->mutable_data(), _outputs[0]->value()->data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToHost));
}

void Variable::clamp(float min, float max)
{
	auto size = _outputs[0]->value()->size();
	VariableClampKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*) _outputs[0]->value()->mutableData(), min, max);
	DF_KERNEL_CHECK();
}

std::string Variable::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.variable(" + _initializer->to_cpp() + ", ";	
	if (_param->variable_param().solver_name().empty())
		cpp += "NULL, ";
	else
		cpp += _param->variable_param().solver_name() + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}