#include "core/node.h"
#include <functional>

#include <glog/logging.h>

Node::Node() : CudaHelper() 
{	
}

Node::Node(deepflow::NodeParam *param) : CudaHelper() {
	_param = param;	
	_name = param->name();
	_scope = param->scope();
}

void Node::createIO() {
	for (int i = 0; i < minNumInputs(); ++i)
		_inputs.push_back(std::make_shared<NodeInput>(shared_from_this(), i));		
	LOG_IF(FATAL, _param->input_size() != _inputs.size()) << name() << " _param->input_size() != minNumInputs() | " << _param->input_size() << " != " << _inputs.size();

	for (int i = 0; i < minNumOutputs(); ++i)
		_outputs.push_back(std::make_shared<NodeOutput>(shared_from_this(), i, name() + std::string("_output_") + std::to_string(i)));		
	LOG_IF(FATAL, _param->output_size() != _outputs.size()) << name() << " _param->output_size() != minNumOutputs()";
}

std::list<std::shared_ptr<Node>> Node::inputNodes() const {
	std::list<std::shared_ptr<Node>> list;
	for (int i = 0; i < _inputs.size(); ++i)
		if (_inputs[i]) {
			auto connectedNode = _inputs[i]->connectedNode();
			if (connectedNode)
				list.push_back(connectedNode);
		}
	return list;
}

void Node::write_values(std::shared_ptr<Tensor> tensor, float alpha, float beta)
{	
	LOG_IF(INFO, _verbose > 2) << tensor->name() << " -> " << _outputs[0]->value()->name();
	auto feed_dim = tensor->dims();
	auto my_dim = _outputs[0]->value()->dims();
	LOG_IF(FATAL, feed_dim != my_dim) << _name << " Forward feed dimension mismatch between dst (" << _outputs[0]->value()->name()  << " - " << _outputs[0]->value()->shape() << ") and src (" << tensor->name()  << " - " << tensor->shape() << ")";
	cpy(_outputs[0]->value()->size(), alpha, tensor->data(), beta, _outputs[0]->value()->mutableData());
}

void Node::write_values(std::initializer_list<float> values)
{
	_outputs[0]->value()->set(values);
}

void Node::write_diffs(std::shared_ptr<Tensor> tensor, float alpha, float beta)
{	
	LOG_IF(INFO, _verbose > 2) << tensor->name() << " -> " << _outputs[0]->diff()->name();	
	auto feed_dim = tensor->dims();
	auto my_dim = _outputs[0]->diff()->dims();
	LOG_IF(FATAL, feed_dim != my_dim) << _name << " Backward feed dimension mismatch between dst (" << _outputs[0]->diff()->name() << " - " << _outputs[0]->diff()->shape() << ") and src (" << tensor->name() << " - " << tensor->shape() << ")";
	cpy(_outputs[0]->diff()->size(), alpha, tensor->data(), beta, _outputs[0]->diff()->mutableData());
}

void Node::write_diffs(std::initializer_list<float> values)
{
	_outputs[0]->diff()->set(values);
}

void Node::reset_gradients()
{
	LOG(FATAL);
}

std::string Node::name() const {
	return _name;	
}

std::string Node::scope() const
{
	return _scope;
}

std::string Node::_input_name_for_cpp(int i) const
{
	auto inputNode = _inputs[i]->connectedNode();
	std::string name = inputNode->name();
	if (inputNode->outputs().size() > 1) {
		int index = 0;
		for (index = 0; index < inputNode->outputs().size(); ++index) {
			if (inputNode->output(index)->name() == _param->input(i))
				break;
		}
		name += "[" + std::to_string(index) + "]";
	}
	return name;
}

std::vector<NodeInputPtr> & Node::inputs() {
	return _inputs;
}

std::vector<NodeOutputPtr> &Node::outputs() {
	return _outputs;
}

NodeInputPtr Node::input(int index) {
	return _inputs[index];
}

NodeOutputPtr Node::output(int index) {
	return _outputs[index];
}

bool Node::isInitialized() const {
	return _initialized;
}

void Node::setInitialized(bool status) {
	_initialized = status;
}

deepflow::NodeParam *Node::param() {
	return _param;
}

void Node::setExecutionContext(ExecutionContextPtr context) {
	_context = context;
	_verbose = _context->debug_level;
}

ExecutionContextPtr Node::executionContext()
{
	return _context;
}

std::list<std::shared_ptr<Node>> Node::outputNodes() const
{
	std::list<std::shared_ptr<Node>> list;
	for (int i = 0; i < _outputs.size(); ++i)
		if (_outputs[i]) {
			for (auto node : _outputs[i]->connectedNodes()) {
					list.push_back(node);
			}
		}
	return list;
}
