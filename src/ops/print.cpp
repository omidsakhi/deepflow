#include "ops/print.h"

Print::Print(const NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_print_param() == false) << "param.has_print_param() == false";
}

int Print::minNumInputs() {
	return _num_inputs;
}

void Print::initForward() {

}

void Print::initBackward() {

}

void Print::forward() {

}

void Print::backward() {

}
