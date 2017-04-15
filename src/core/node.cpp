#include "core/node.h"
#include "core/node_observer.h"

#include <glog/logging.h>

Node::Node(const NodeParam &param) : CudaHelper() {
	_param = param;	
	_name = param.name();
	_visited = false;
}

void Node::createIO() {
	for (int i = 0; i < minNumInputs(); ++i)
		_inputs.push_back(std::make_shared<NodeInput>(shared_from_this(), i));		
	//for (int i = _param.input_size(); i < minNumInputs(); ++i)
	//	_param.add_input();
	LOG_IF(FATAL, _param.input_size() != _inputs.size()) << name() << " _param.input_size() != minNumInputs()";

	for (int i = 0; i < minNumOutputs(); ++i)
		_outputs.push_back(std::make_shared<NodeOutput>(shared_from_this(), i, name() + std::string("_output_") + std::to_string(i)));		
	//for (int i = _param.output_size(); i < minNumOutputs(); ++i)
	//	_param.add_output();
	LOG_IF(FATAL, _param.output_size() != _outputs.size()) << name() << " _param.output_size() != minNumOutputs()";
}

void Node::setVisited(bool state) {
	_visited = state;
}

std::list<std::shared_ptr<Node>> Node::sourceNodes() const {
	std::list<std::shared_ptr<Node>> list;
	for (int i = 0; i < _inputs.size(); ++i)
		if (_inputs[i]) {
			auto connectedNode = _inputs[i]->connectedNode();
			if (connectedNode)
				list.push_back(connectedNode);
		}
	return list;
}

void Node::traverse(NodeObserver *observer, TraverseOrder order, bool visit_condition) {
	if (_visited == visit_condition)
		return;
	if (_context && includePhase(_context->phase) == false)
		return;
	if (order == TraverseOrder::PreOrder)
		observer->apply(shared_from_this());
	for (auto node : sourceNodes())
		node->traverse(observer, order, visit_condition);
	if (order == TraverseOrder::PostOrder)
		observer->apply(shared_from_this());
	_visited = visit_condition;
}

void Node::forward() {}

void Node::backward() {}

std::string Node::name() const {
	return _name;	
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

NodeParam &Node::param() {
	return _param;
}

bool Node::includePhase(const std::string &phase) {
	if (_param.phase_size() == 0)
		return true;
	for (int i = 0; i < _param.phase_size(); ++i)
		if (_param.phase(i) == phase)
			return true;
	return false;
}

void Node::setExecutionContext(ExecutionContextPtr context) {
	_context = context;
}
