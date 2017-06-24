
#include "nodes/variable.h"
#include "core/initializer.h"

#include <string>
#include <iostream>

#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

Variable::Variable(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_variable_param() == false) << "param.has_variable_param() == false";
	_initializer = initializer;			
}

void Variable::initForward() {	
	_initializer->init();
	_outputs[0]->initValue(_initializer->dims());
	LOG(INFO) << "Variable " << _name << " - " << _outputs[0]->value()->shape();
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
}

void Variable::initBackward() {	
	_outputs[0]->initDiff();
}

inline void Variable::forward() {
	
}

inline void Variable::backward() {

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

Node::BackwardType Variable::backwardType()
{
	return _param->mutable_variable_param()->solver_name().empty() ? NEVER_BACKWARD : ALWAYS_BACKWARD;
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