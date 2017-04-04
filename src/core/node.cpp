#include "core/node.h"
#include "core/node_observer.h"

#include <glog/logging.h>

Node::Node(NodeParam param) : CudaHelper() {
    _param = param;	
    _name = param.name();
    LOG(INFO) << "Creating Node (name: " << _name << " )";
    _visited = false;
}

void Node::createIO() {
    for (int i = 0; i < minNumInputs(); ++i)
        _inputs.push_back(std::make_shared<InputTerminal>(shared_from_this(), i, ""));
    for (int i = 0; i < minNumOutputs(); ++i)
        _outputs.push_back(std::make_shared<OutputTerminal>(shared_from_this(), i, ""));
}

void Node::setVisited(bool state) {
    _visited = state;
}

void Node::traverse(NodeObserver *observer, TraverseOrder order, bool visit_condition) {
    if (_visited == visit_condition)
        return;
    if (order == TraverseOrder::PreOrder)
        observer->apply(shared_from_this());
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i]) {
            auto connectedNode = _inputs[i]->connectedNode();
            if (connectedNode)
				connectedNode->traverse(observer, order, visit_condition);
        }			
    if (order == TraverseOrder::PostOrder)
        observer->apply(shared_from_this());
    _visited = visit_condition;
}

void Node::forward() {}

void Node::backward() {}

std::string Node::name() {
    return _name;
}

std::vector<std::shared_ptr<InputTerminal>> & Node::inputs() {
    return _inputs;
}

std::vector<std::shared_ptr<OutputTerminal>> &Node::outputs() {
    return _outputs;
}

std::shared_ptr<InputTerminal> Node::input(int index) {
    return _inputs[index];
}

std::shared_ptr<OutputTerminal> Node::output(int index) {
    return _outputs[index];
}

bool Node::isInitialized() const {
	return _initialized;
}

void Node::setInitialized(bool status) {
	_initialized = status;
}