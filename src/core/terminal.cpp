#include "core/terminal.h"

#include "core/node.h"

Terminal::Terminal(std::shared_ptr<Node> parentNode, int index, std::string name, TerminalType type) {
	_parentNode = parentNode;
	_index = index;
	_name = name;	
	_type = type;
}

TerminalType Terminal::type() {
	return _type;
}

void InputTerminal::connect(std::shared_ptr<OutputTerminal> terminal) {
	_terminal = terminal;
}

std::shared_ptr<Node> Terminal::parentNode() {
	return _parentNode;
}
std::shared_ptr<Node> Terminal::otherNode() {
	if (_terminal)
		return _terminal->parentNode();
	return 0;
}
InputTerminal::InputTerminal(std::shared_ptr<Node> parentNode, int index, std::string name) : Terminal(parentNode, index, name, TerminalType::Input) {

}

OutputTerminal::OutputTerminal(std::shared_ptr<Node> parentNode, int index, std::string name) : Terminal(parentNode, index, name, TerminalType::Output) {
}

std::shared_ptr<Tensor> InputTerminal::value() {	
	return _terminal->value();
}

std::shared_ptr<Tensor> InputTerminal::diff() {	
	return _terminal->diff();
}

std::shared_ptr<Tensor> OutputTerminal::value() {	
	return _value;
}

std::shared_ptr<Tensor> OutputTerminal::diff() {	
	return _diff;
}

void OutputTerminal::initValue(std::array<int, 4> dims, Tensor::TensorType type) {
	_value = std::make_shared<Tensor>(dims,type);
}

void OutputTerminal::initDiff() {
	_diff = std::make_shared<Tensor>(_value->dims(),_value->type());
}

void OutputTerminal::cpyValueToDiff() {
	LOG_IF(FATAL, cudaMemcpy(_diff->mutableData(), _value->data(), _value->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0) << "cudaMemcpy [FAILED]";
}

std::array<int, 4> InputTerminal::dims() {
	return _terminal->value()->dims();
}

std::array<int, 4> OutputTerminal::dims() {
	return _value->dims();
}