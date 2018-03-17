#include "core/terminal.h"

#include "core/node.h"

Terminal::Terminal(std::shared_ptr<Node> parentNode, int index, TerminalType type) {
	_parentNode = parentNode;
	_index = index;	
	_reader_type = type;
}

const TerminalType& Terminal::type() const {
	return _reader_type;
}

const std::string& NodeInput::name() const {
	return _connected_terminal->name();
}

void NodeInput::connectTerminal(std::shared_ptr<Terminal> terminal)
{
	LOG_IF(FATAL, _connected_terminal != NULL) << "Input node already has a terminal.";
	_connected_terminal = terminal;
}

void NodeInput::disconnect()
{
	_connected_terminal = nullptr;
}

std::shared_ptr<Node> NodeInput::connectedNode() const
{
	if (_connected_terminal)
		return _connected_terminal->parentNode();
	return NULL;
}

std::shared_ptr<Terminal> NodeInput::connectedTerminal() const
{
	return _connected_terminal;
}

const std::string& NodeOutput::name() const {
	return _name;
}

void NodeOutput::connectTerminal(std::shared_ptr<Terminal> terminal)
{
	_connected_terminals.insert(terminal);
}

void NodeOutput::disconnect()
{
	_connected_terminals.clear();
}

std::set<std::shared_ptr<Node>> NodeOutput::connectedNodes() const
{
	std::set<std::shared_ptr<Node>> list;
	for (auto t : _connected_terminals)
		list.insert(t->parentNode());
	return list;
}

std::set<std::shared_ptr<Terminal>> NodeOutput::connectedTerminals() const
{
	return _connected_terminals;
}

void NodeOutput::setEnabled(bool status)
{
	_enabled = status;
}

bool NodeOutput::enabled() const
{
	return _enabled;
}

const int& Terminal::index() const {
	return _index;
}

void NodeInput::connect(std::shared_ptr<NodeOutput> terminal) {
	_connected_terminal = terminal;
	if (_parentNode)
		_parentNode->param()->set_input(_index, terminal->name());
	_connected_terminal->parentNode()->param()->set_output(_connected_terminal->index(), terminal->name());
	terminal->connectTerminal(shared_from_this());	
	LOG_IF(FATAL, _connected_terminal->parentNode()->param()->input_size() != _connected_terminal->parentNode()->inputs().size()) << _connected_terminal->parentNode()->name() << " _param->input_size() != minNumInputs()";
	LOG_IF(FATAL, _connected_terminal->parentNode()->param()->output_size() != _connected_terminal->parentNode()->outputs().size()) << _connected_terminal->parentNode()->name() << " _param->output_size() != minNumOutputs()";
}

std::shared_ptr<Node> Terminal::parentNode() const {
	return _parentNode;
}

NodeInput::NodeInput(std::shared_ptr<Node> parentNode, int index) : Terminal(parentNode, index, TerminalType::Input) {
}

NodeOutput::NodeOutput(std::shared_ptr<Node> parentNode, int index, const std::string &name) : Terminal(parentNode, index, TerminalType::Output) {
	_name = name;
}

std::shared_ptr<Tensor> NodeInput::value() {	
	return _connected_terminal->value();
}

std::shared_ptr<Tensor> NodeInput::diff() {
	if (_connected_terminal)
		return _connected_terminal->diff();
	else
		return nullptr;
}

std::shared_ptr<Tensor> NodeOutput::value() {	
	return _value;
}

std::shared_ptr<Tensor> NodeOutput::diff() {	
	return _diff;
}

void NodeOutput::initValue(std::array<int, 4> dims, Tensor::TensorType type) {
	LOG_IF(FATAL, _value != nullptr) << "_value != nullptr";
	_value = std::make_shared<Tensor>(dims,type, _name + "_v");
}

void NodeOutput::initValue(std::array<int, 4> dims, std::shared_ptr<Tensor> tensor)
{
	LOG_IF(FATAL, _value != nullptr) << "_value != nullptr";
	_value = std::make_shared<Tensor>(dims, tensor, _name + "_v");
}

void NodeOutput::initDiff(bool reset)
{
	LOG_IF(FATAL, _value == nullptr) << "_value == nullptr";
	LOG_IF(FATAL, _diff != nullptr) << "_diff != nullptr";
	_diff = std::make_shared<Tensor>(_value->dims(), _value->type(), _name + "_d", reset);
}

void NodeOutput::initDiff(std::array<int, 4> dims, std::shared_ptr<Tensor> tensor)
{
	LOG_IF(FATAL, _value == nullptr) << "_value == nullptr";
	LOG_IF(FATAL, _diff != nullptr) << "_diff != nullptr";
	LOG_IF(FATAL, dims != _value->dims()) << "dims != _value->dims()";
	_diff = std::make_shared<Tensor>(dims, tensor, _name + "_d");
}

void NodeOutput::resetDiff(cudaStream_t stream)
{
	if (_diff) {
		if (stream) {
			DF_NODE_CUDA_CHECK(cudaMemsetAsync(_diff->mutableData(), 0, _diff->sizeInBytes(), stream));
		}
		else {
			DF_NODE_CUDA_CHECK(cudaMemset(_diff->mutableData(), 0, _diff->sizeInBytes()));
		}		
	}
}

void NodeOutput::resetValue()
{
	DF_NODE_CUDA_CHECK(cudaMemset(_value->mutableData(), 0, _value->sizeInBytes()));
}

void NodeOutput::cpyValueToDiff() {
	DF_NODE_CUDA_CHECK(cudaMemcpy(_diff->mutableData(), _value->data(), _value->sizeInBytes(), cudaMemcpyDeviceToDevice));	
}

std::array<int, 4> NodeInput::dims() {
	return _connected_terminal->value()->dims();
}

std::array<int, 4> NodeOutput::dims() {
	return _value->dims();
}

void NodeOutput::feed(std::shared_ptr<NodeOutput> t) {
	LOG_IF(FATAL, t->value()->sizeInBytes() != value()->sizeInBytes()) << "Size mismatch between terminals: " << _name << " and " << t->name();
	DF_NODE_CUDA_CHECK(cudaMemcpy(value()->mutableData(), t->value()->data(), value()->sizeInBytes(), cudaMemcpyDeviceToDevice));	
}