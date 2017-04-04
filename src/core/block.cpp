#include "core/block.h"
#include "observers/forward.h"
#include "observers/backward.h"

#include <glog/logging.h>

Block::Block(std::initializer_list<std::shared_ptr<InputTerminal>> inputs, std::initializer_list<std::shared_ptr<OutputTerminal>> outputs, NodeParam param) : Node(param) {
	LOG_IF(FATAL, param.has_block_param() == false) << "param.has_block_param() [FAILED]";	
	for (auto inputTerminal : inputs)
		_inputs.push_back(inputTerminal);		
	for (auto outputTerminal : outputs)
		_outputs.push_back(outputTerminal);	
}

void Block::initForward() {
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	//for (auto node : nodes)
	//	node->recursiveInitForward();
}

void Block::initBackward() {
	//InitBackwardObserver _;
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	//for (auto node : nodes)
	//	node->recursiveInitBackward();
}

void Block::forward() {
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	//for (auto node : nodes)
	//	node->recursiveForward();
}

void Block::backward() {
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	//for (auto node : nodes)
	//	node->recursiveBackward();
}

void Block::traverse(NodeObserver *observer, TraverseOrder order, bool visit_condition) {
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	for (auto node : nodes)
		node->traverse(observer, order, visit_condition);
}

void Block::recursiveResetVisit() {
	std::set<std::shared_ptr<Node>> nodes;
	for (int i = 0; i < _outputs.size(); ++i)
		nodes.insert(_outputs[i]->node());
	//for (auto node : nodes)
	//	node->recursiveResetVisit();
}