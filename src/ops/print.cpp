#include "ops/print.h"

Print::Print(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_print_param() == false) << "param.has_print_param() == false";
	const PrintParam &printParam = _param.print_param();
	_num_inputs = printParam.num_inputs();
}

int Print::minNumInputs() {	
	return _num_inputs;
}

void Print::initForward() {
	const PrintParam &printParam = _param.print_param();
	_raw_message = printParam.message();
}

void Print::initBackward() {

}

void Print::forward() {
	std::string message = _raw_message;
	for (int i = 0; i < _inputs.size(); ++i) {
		size_t start_pos = message.find("{" + std::to_string(i) + "}");
		if (start_pos == std::string::npos)
			continue;
		message.replace(start_pos, 3, _inputs[i]->value()->toString());		
	}
	std::cout << message;
}

void Print::backward() {

}
