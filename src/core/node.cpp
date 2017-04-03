#include "core/node.h"
#include "core/node_observer.h"

#include <glog/logging.h>

Node::Node(NodeParam param) : CudaHelper() {
    _param = param;	
    _name = param.name();
    LOG(INFO) << "Creating Node (name: " << _name << " )";
    _visited = false;
}

void Node::init() {
    for (int i = 0; i < minNumInputs(); ++i)
        _inputs.push_back(std::make_shared<InputTerminal>(shared_from_this(), i, ""));
    for (int i = 0; i < minNumOutputs(); ++i)
        _outputs.push_back(std::make_shared<OutputTerminal>(shared_from_this(), i, ""));
}

void Node::setVisited(bool state) {
    _visited = state;
}

void Node::traverse(NodeObserver *observer, TraverseOrder order) {
    if (_visited == true)
        return;
    if (order == TraverseOrder::PreOrder)
        observer->apply(this);
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i]) {
            auto node = _inputs[i]->otherNode();
            if (node)
                node->traverse(observer, order);
        }			
    if (order == TraverseOrder::PostOrder)
        observer->apply(this);
    _visited = true;
}

void Node::recursiveResetVisit() {
    if (_visited == false)
        return;
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i])
        {
            auto node = _inputs[i]->otherNode();
            if (node) node->recursiveResetVisit();
        }
    _visited = false;
}

void Node::recursiveForward() {
    if (_visited == true)
        return;
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i])
        {
            auto node = _inputs[i]->otherNode();
            if (node) node->recursiveForward();			
        }	
    forward();	
    _visited = true;
}

void Node::recursiveBackward() {
    if (_visited == true)
        return;	
    backward();
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i])
        {
            auto node = _inputs[i]->otherNode();
            if (node) node->recursiveBackward();
        }	
    _visited = true;
}

void Node::recursiveInitForward() {
    if (_visited == true)
        return;	
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i])
        {
            auto node = _inputs[i]->otherNode();
            if (node) node->recursiveInitForward();
        }
    initForward();
    _visited = true;
}

void Node::recursiveInitBackward() {
    if (_visited == true)
        return;
    for (int i = 0; i < _inputs.size(); ++i)
        if (_inputs[i])
        {
            auto node = _inputs[i]->otherNode();
            if (node) node->recursiveInitBackward();
        }	
    initBackward();
    _visited = true;
}

void Node::deinit() {

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