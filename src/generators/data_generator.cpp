#include "generators/data_generator.h"
#include "nodes/variable.h"
#include "core/initializer.h"
#include "core/common_cu.h"

DataGenerator::DataGenerator(std::shared_ptr<Initializer> initializer, const deepflow::NodeParam &param) : Generator(param), Variable(initializer,param) {
	LOG_IF(FATAL, param.generator_param().has_data_generator_param() == false) << "param.generator_param().has_data_generator_param() == false";
	_no_solver = param.variable_param().solver_name().empty();
	_num_total_samples = param.generator_param().data_generator_param().num_samples();
	_batch_size = param.variable_param().init_param().tensor_param().dims(0);
	LOG_IF(FATAL, _batch_size < 1) << "Data generator batch size must be more than 0";
	LOG_IF(FATAL, _num_total_samples % _batch_size != 0) << "Number of total samples " << _num_total_samples << " must be dividable by the batch size " << _batch_size;
	_num_batches = _num_total_samples / _batch_size;
	_last_batch = (_current_batch == (_num_batches - 1));
}

void DataGenerator::nextBatch() {
	if (_no_solver) {
		_initializer->apply(this);		
		_last_batch = (_current_batch >= (_num_batches - 1));
		if (_last_batch) {
			_current_batch = 0;			
		}
		else {			
			_current_batch++;
		}		
	}
	LOG_IF(INFO, (_context && _context->debug_level > 2)) << "MNIST " << _name << " - BATCH @ " << _current_batch;
}

void DataGenerator::initForward() {
	_initializer->init();
	
	_outputs[0]->initValue(_initializer->dims());
	LOG(INFO) << "Initializing Data Generator " << _name << " - " << _outputs[0]->value()->shape();
	if (_param.variable_param().has_weights()) {
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _param.variable_param().weights().weight().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else if (_initializer->param().has_init_data()) {
		DF_NODE_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _initializer->param().init_data().weight().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else {
		_initializer->apply(this);
		for (int i = 0; i < _outputs[0]->value()->size(); ++i)
			_param.mutable_variable_param()->mutable_init_param()->mutable_init_data()->add_weight(0);
		DF_NODE_CUDA_CHECK(cudaMemcpy(_param.mutable_variable_param()->mutable_init_param()->mutable_init_data()->mutable_weight()->mutable_data(), _outputs[0]->value()->data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToHost));
	}	
}

void DataGenerator::initBackward() {
	_outputs[0]->initDiff();
}

void DataGenerator::forward() {
	
}

void DataGenerator::backward() {
	
}

bool DataGenerator::isLastBatch() {
	return _last_batch;
}

std::string DataGenerator::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.data_generator(" + _initializer->to_cpp() + ", ";
	cpp += std::to_string(_num_total_samples) + ", ";
	cpp += (_no_solver ? "NULL" : _param.variable_param().solver_name()) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
