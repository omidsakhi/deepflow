#pragma once

#include "core/export.h"

#include <memory>
#include <vector>

#include "cudnn.h"

#include "core/tensor.h"
#include "core/cuda_helper.h"

#include <list>

#include <initializer_list>
#include "core/terminal.h"

#include <string>
#include <vector>

#include <iostream>

#include "proto/deepflow.pb.h"

class NodeObserver;

enum TraverseOrder {
	PreOrder,
	PostOrder
};

enum NodePrintType {
	Value,
	Diff
};

class DeepFlowDllExport Node : public std::enable_shared_from_this<Node>, public CudaHelper {
public:
	Node(NodeParam param);
	void init();
	virtual void initForward() = 0;
	virtual void initBackward() = 0;
	virtual int minNumInputs() = 0;
	virtual int minNumOutputs() = 0;
	virtual void deinit();
	virtual void forward();
	virtual void backward();	
	std::string name();		
	virtual void traverse(NodeObserver *observer, TraverseOrder order);
	virtual void recursiveResetVisit();
	virtual void recursiveForward();
	virtual void recursiveBackward();
	virtual void recursiveInitForward();
	virtual void recursiveInitBackward();
	template<typename T> 
	void recursiveNodeFinder(std::list<std::shared_ptr<T>> &list);
	void setVisited(bool state);
	std::vector<std::shared_ptr<InputTerminal>> &inputs();
	std::vector<std::shared_ptr<OutputTerminal>> &outputs();
	std::shared_ptr<InputTerminal> input(int index);
	std::shared_ptr<OutputTerminal> output(int index);
protected:	
	std::vector<std::shared_ptr<InputTerminal>> _inputs;
	std::vector<std::shared_ptr<OutputTerminal>> _outputs;
	std::string _name;	
	bool _visited;
	NodeParam _param;	
};

template<typename T>
void Node::recursiveNodeFinder(std::list<std::shared_ptr<T>> &list) {
	if (_visited == true)
		return;
	if (dynamic_cast<T*>(this)) {
		list.push_back(std::dynamic_pointer_cast<T>(shared_from_this()));
	}
	for (int i = 0; i < _inputs.size(); ++i)
		if (_inputs[i]) {
			auto node = _inputs[i]->otherNode();
			if (node)
				node->recursiveNodeFinder<T>(list);
		}
	_visited = true;
}
