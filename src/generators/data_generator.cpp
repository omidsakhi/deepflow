#include "generators/data_generator.h"
#include "nodes/variable.h"
#include "core/initializer.h"
#include "core/common_cu.h"

DataGenerator::DataGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param) : Variable(initializer,param) {
	LOG_IF(FATAL, param->has_data_generator_param() == false) << "param->has_data_generator_param() == false";
	_no_solver = param->variable_param().solver_name().empty();	
	_batch_size = param->variable_param().init_param().tensor_param().dims(0);
	LOG_IF(FATAL, _batch_size < 1) << "Data generator batch size must be more than 0";	
}

bool DataGenerator::is_generator()
{	
	return _no_solver;
}

void DataGenerator::forward() {
	if (_no_solver) {
		_initializer->apply(this);
	}	
}
std::string DataGenerator::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.data_generator(" + _initializer->to_cpp() + ", ";
	cpp += (_no_solver ? "NULL" : _param->variable_param().solver_name()) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}