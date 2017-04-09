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

const std::string& NodeInput::name() const {
	return _terminal->name();
}

const std::string& NodeOutput::name() const {
	return _name;
}

const int& Terminal::index() const {
	return _index;
}

void NodeInput::connect(std::shared_ptr<NodeOutput> terminal) {
	_terminal = terminal;
	_parentNode->param().set_input(_index, terminal->name());
	_terminal->node()->param().set_output(_terminal->index(), terminal->name());	
	terminal->setTerminal(shared_from_this());
}

void Terminal::setTerminal(std::shared_ptr<Terminal> terminal) {
	_terminal = terminal;
}

std::shared_ptr<Node> Terminal::node() const {
	return _parentNode;
}
std::shared_ptr<Terminal> Terminal::connectedTerminal() const {
	if (_terminal)
		return _terminal;
	return 0;
}

std::shared_ptr<Node> Terminal::connectedNode() const {
	if (_terminal)
		return _terminal->node();
	return 0;
}
NodeInput::NodeInput(std::shared_ptr<Node> parentNode, int index) : Terminal(parentNode, index, TerminalType::Input) {
}

NodeOutput::NodeOutput(std::shared_ptr<Node> parentNode, int index, const std::string &name) : Terminal(parentNode, index, TerminalType::Output) {
	_name = name;
}

std::shared_ptr<Tensor> NodeInput::value() {	
	return _terminal->value();
}

std::shared_ptr<Tensor> NodeInput::diff() {	
	return _terminal->diff();
}

std::shared_ptr<Tensor> NodeOutput::value() {	
	return _value;
}

std::shared_ptr<Tensor> NodeOutput::diff() {	
	return _diff;
}

void NodeOutput::initValue(std::array<int, 4> dims, Tensor::TensorType type) {
	_value = std::make_shared<Tensor>(dims,type);
}

void NodeOutput::initDiff() {
	_diff = std::make_shared<Tensor>(_value->dims(),_value->type());
}

void NodeOutput::cpyValueToDiff() {
	LOG_IF(FATAL, cudaMemcpy(_diff->mutableData(), _value->data(), _value->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0) << "cudaMemcpy [FAILED]";
}

std::array<int, 4> NodeInput::dims() {
	return _terminal->value()->dims();
}

std::array<int, 4> NodeOutput::dims() {
	return _value->dims();
}

void NodeOutput::feed(std::shared_ptr<NodeOutput> t) {
	LOG_IF(FATAL, t->value()->sizeInBytes() != value()->sizeInBytes()) << "Size mismatch between terminals: " << _name << " and " << t->name();
	LOG_IF(FATAL, cudaMemcpy(value()->mutableData(), t->value()->data(), value()->sizeInBytes(), cudaMemcpyDeviceToDevice) != 0) << "cudaMemcpy [FAILED]";
}