#include "nodes/print.h"

Print::Print(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_print_param() == false) << "param.has_print_param() == false";
	auto printParam = _param->print_param();
	_num_inputs = printParam.num_inputs();
	_print_time = printParam.print_time();
	_print_type = printParam.print_type();
}

int Print::minNumInputs() {	
	return _num_inputs;
}

void Print::init() {
	auto printParam = _param->print_param();
	_raw_message = printParam.message();
}

void Print::forward() {
	if (_print_time == deepflow::ActionTime::END_OF_EPOCH && _context->last_batch == false)
		return;
	std::string message = _raw_message;
	size_t start_pos = message.find("%{}");
	if (start_pos != std::string::npos) {
		float a, b;
		if (_print_type == deepflow::ActionType::DIFFS) {
			a = atof(_inputs[0]->diff()->toString().c_str());
			b = atof(_inputs[1]->diff()->toString().c_str());
		}
		else {
			a = atof(_inputs[0]->value()->toString().c_str());
			b = atof(_inputs[1]->value()->toString().c_str());
		}
		message.replace(start_pos, 3, std::to_string(a / b * 100.0f));
	}

	if (_context) {
		start_pos = message.find("{ep}");
		if (start_pos != std::string::npos) {
			message.replace(start_pos, 4, std::to_string(_context->current_epoch));
		}
		start_pos = message.find("{it}");
		if (start_pos != std::string::npos) {
			message.replace(start_pos, 4, std::to_string(_context->current_iteration));
		}
	}

	for (int i = 0; i < _inputs.size(); ++i) {
		size_t start_pos = message.find("{" + std::to_string(i) + "}");
		if (start_pos == std::string::npos)
			continue;
		if (_print_type == deepflow::ActionType::DIFFS)
			message.replace(start_pos, 3, _inputs[i]->diff()->toString());
		else
			message.replace(start_pos, 3, _inputs[i]->value()->toString());
	}
	
	std::cout << message;
}

void Print::backward() {

}

std::string Print::to_cpp() const
{	
	std::string inputs = _input_name_for_cpp(0);
	for (int i = 1; i < _num_inputs; ++i) {
		inputs += ", " + _input_name_for_cpp(i);
	}	

	std::string cpp = "df.print( {" + inputs + "} , ";	
	std::string escaped_raw_message = _raw_message;
	for (int i=0; i < escaped_raw_message.size(); ++i) 
		if (escaped_raw_message[i] == '\n') {
			escaped_raw_message[i] = 'n';
		}		
	cpp += "\"" + escaped_raw_message + "\", ";
	if (_print_time == deepflow::ActionTime::END_OF_EPOCH) {
		cpp += "Print::END_OF_EPOCH, ";
	}
	else {
		cpp += "Print::EVERY_PASS, ";
	}
	if (_print_type == deepflow::ActionType::DIFFS) {
		cpp += "Print::DIFFS, ";
	}
	else {
		cpp += "Print::VALUES, ";
	}
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
