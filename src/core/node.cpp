#include "core/node.h"
#include <functional>

#include <glog/logging.h>

Node::Node() : CudaHelper() 
{	
}

Node::Node(deepflow::NodeParam *param) : CudaHelper() {
	_param = param;	
	_name = param->name();	
}

void Node::createIO() {
	for (int i = 0; i < minNumInputs(); ++i)
		_inputs.push_back(std::make_shared<NodeInput>(shared_from_this(), i));		
	LOG_IF(FATAL, _param->input_size() != _inputs.size()) << name() << " _param->input_size() != minNumInputs() | " << _param->input_size() << " != " << _inputs.size();

	for (int i = 0; i < minNumOutputs(); ++i)
		_outputs.push_back(std::make_shared<NodeOutput>(shared_from_this(), i, name() + std::string("_output_") + std::to_string(i)));		
	LOG_IF(FATAL, _param->output_size() != _outputs.size()) << name() << " _param->output_size() != minNumOutputs()";
}

void Node::setVisited(bool state) {
	_visited = state;
}

void Node::resetGradients()
{	
	LOG_IF(INFO, _verbose > 2) << "RESET GRADIENTS " << _name;
	for (auto output: _outputs)
		output->resetDiff();
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

void Node::_unvisit() {
	if (_visited == false)
		return;
	_visited = false;
	for (auto node : inputNodes())
		node->_unvisit();
}

void Node::_resolve_propagation() {
	if (_visited == true)
		return;
	_visited = true;	
	LOG_IF(INFO, _verbose > 3) << "RESOLVING BP - VISITING " << _name;
	if (backwardType() == ALWAYS_BACKWARD) {				
		LOG_IF(INFO, _verbose > 3) << "RESOLVING BP - ALWAYS BACKWARD " << _name;
		setShouldBackward(true);
	}
	else if (backwardType() == NEVER_BACKWARD) {
		LOG_IF(INFO, _verbose > 3) << "RESOLVING BP - NEVER BACKWARD " << _name;
		setShouldBackward(false);
	}
	else {
		bool _should_backward = false;
		auto list = inputNodes();
		for (auto node : list) {
			LOG_IF(INFO, _verbose > 3) << "RESOLVING BP - INSPECTING INPUT " << node->name();
			node->_resolve_propagation();
			if (node->_propagate_back == true) {
				_should_backward = true;				
				break;
			}
		}
		LOG_IF(INFO, _verbose > 3) << "RESOLVING BP - " << _name << (_should_backward?" SHOULD PROPAGATE." : " SHOULD **NOT** PROPAGATE.");
		setShouldBackward(_should_backward);
	}	
	for (auto node : inputNodes()) {
		node->_resolve_propagation();
	}
}

bool Node::propagateBack() const
{
	return _propagate_back;
}

void Node::setShouldBackward(bool state)
{
	_propagate_back = state;	
}

void Node::write_values(std::shared_ptr<Tensor> tensor, float alpha, float beta)
{	
	LOG_IF(INFO, _verbose > 2) << "FEEDING VALUES TO " << _name;		
	auto feed_dim = tensor->dims();
	auto my_dim = _outputs[0]->value()->dims();
	LOG_IF(FATAL, feed_dim != my_dim) << "Forward feed dimension mismatch between " << _name << " (src - " << _outputs[0]->value()->shape() << " ) and dst - " << tensor->shape();
	cpy(_outputs[0]->value()->size(), alpha, tensor->data(), beta, _outputs[0]->value()->mutableData());
}

void Node::write_diffs(std::shared_ptr<Tensor> tensor, float alpha, float beta)
{	
	LOG_IF(INFO, _verbose > 2) << "FEEDING GRADIENTS FROM TO " << _name;	
	auto feed_dim = tensor->dims();
	auto my_dim = _outputs[0]->diff()->dims();
	LOG_IF(FATAL, feed_dim != my_dim) << "Backward feed dimension mismatch between " << _name << " (src - " << _outputs[0]->value()->shape() << " ) and dst - " << tensor->shape();
	cpy(_outputs[0]->diff()->size(), alpha, tensor->data(), beta, _outputs[0]->diff()->mutableData());
}

void Node::_traverse_up(std::function<void(Node*)> fun, TraverseOrder order, bool visit_condition)
{
	if (_visited == visit_condition)
		return;
	if (_context && includePhase(_context->phase) == false)
		return;
	if (order == PRE_ORDER)
		fun(this);
	for (auto node : inputNodes())
		node->_traverse_up(fun, order, visit_condition);
	if (order == POST_ORDER)
		fun(this);
	_visited = visit_condition;
}

void Node::_traverse_down(std::function<void(Node*)> fun, TraverseOrder order, bool visit_condition)
{
	if (_visited == visit_condition)
		return;
	if (_context && includePhase(_context->phase) == false)
		return;
	if (order == PRE_ORDER)
		fun(this);
	for (auto node : outputNodes())
		node->_traverse_down(fun, order, visit_condition);
	if (order == POST_ORDER)
		fun(this);
	_visited = visit_condition;
}

void Node::_forward() {	
	if (_visited == true) {
		LOG_IF(INFO, _verbose > 2) << "SKIP FORWARD" << _name << " DUE TO VISITED = true ";
		return;
	}
	if (_context && includePhase(_context->phase) == false) {
		LOG_IF(INFO, _verbose > 2) << "SKIP FORWARD " << _name << " DUE TO PHASE";
		return;
	}
	_visited = true;	
	for (auto node : inputNodes()) {
		LOG_IF(INFO, _verbose > 3) << "FROM " << _name << " GOING TO FORWARD " << node->name();
		node->_forward();
	}	
	LOG_IF(INFO, _verbose > 2) << "FORWARD " << _name;
	forward();
	if (_verbose > 3) {
		for (auto output : _outputs) {
			if (output->value()) {
				int res = output->value()->isValid();
				LOG_IF(FATAL, res != 0) << ((res == -1) ? "INF" : "NAN");
			}
		}
	}
}

void Node::_backward() {	
	if (_visited == true) {
		LOG_IF(INFO, _verbose > 2) << "SKIP BACKWARD " << _name << " DUE TO VISITED = true ";
		return;
	}
	if (_context && includePhase(_context->phase) == false) {
		LOG_IF(INFO, _verbose > 2) << "SKIP BACKWARD " << _name << " DUE TO PHASE";
		return;
	}
	_visited = true;
	for (auto node : outputNodes()) {
		LOG_IF(INFO, _verbose > 3) << "FROM " << _name << " GOING TO BACKWARD " << node->name();
		node->_backward();
	}
	if (_propagate_back) {
		LOG_IF(INFO, _verbose > 2) << "BACKWARD " << _name;
		backward();
		if (_verbose > 3) {
			for (auto input : _inputs) {
				if (input->diff()) {
					int res = input->diff()->isValid();
					LOG_IF(FATAL, res != 0) << ((res == -1) ? "INF" : "NAN");
				}
			}
		}
	}
}

std::string Node::name() const {
	return _name;	
}

std::string Node::_to_cpp_phases() const
{
	std::string phases;
	if (_param->phase_size() > 0)
		phases += "\"" + _param->phase(0) + "\"";
	for (int i = 1; i < _param->phase_size(); ++i) {
		phases = phases + ", " + "\"" + _param->phase(i) + "\"";
	}
	return phases;
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

int Node::numConnectedOutputs()
{
	int sum = 0;
	for (auto terminal: _outputs) {
		sum += terminal->connectedNodes().size();
	}
	return sum;
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

bool Node::includePhase(const std::string &phase) {
	if (phase.empty())
		return true;
	if (_param->phase_size() == 0)
		return true;
	for (int i = 0; i < _param->phase_size(); ++i)
		if (_param->phase(i) == phase)
			return true;
	return false;
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
			for (auto node : _outputs[i]->connectedNodes())
				list.push_back(node);
		}
	return list;
}
