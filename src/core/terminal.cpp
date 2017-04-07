#include "core/terminal.h"

#include "core/node.h"

Terminal::Terminal(std::shared_ptr<Node> parentNode, int index, TerminalType type) {
	_parentNode = parentNode;
	_index = index;	
	_type = type;
}

const TerminalType& Terminal::type() const {
	return _type;
}

const std::string& InputTerminal::name() const {
	return _terminal->name();
}

const std::string& OutputTerminal::name() const {
	return _name;
}

const int& Terminal::index() const {
	return _index;
}

void InputTerminal::connect(std::shared_ptr<OutputTerminal> terminal) {
	_terminal = terminal;
	_parentNode->param().set_input(_index, terminal->name());
	_terminal->node()->param().set_output(_terminal->index(), terminal->name());
}

std::shared_ptr<Node> Terminal::node() const {
	return _parentNode;
}
std::shared_ptr<Node> Terminal::connectedNode() const {
	if (_terminal)
		return _terminal->node();
	return 0;
}
InputTerminal::InputTerminal(std::shared_ptr<Node> parentNode, int index) : Terminal(parentNode, index, TerminalType::Input) {
}

OutputTerminal::OutputTerminal(std::shared_ptr<Node> parentNode, int index, const std::string &name) : Terminal(parentNode, index, TerminalType::Output) {
	_name = name;
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

void OutputTerminal::feed(std::shared_ptr<OutputTerminal> t) {
	LOG_IF(FATAL, t->value()->sizeInBytes() != value()->sizeInBytes()) << "Size mismatch between terminals: " << _name << " and " << t->name();
	LOG_IF(FATAL, cudaMemcpy(value()->mutableData(), t->value()->data(), value()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0) << "cudaMemcpy [FAILED]";
}