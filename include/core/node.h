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
	Node(const NodeParam &param);
	void createIO();
	virtual void initForward() = 0;
	virtual void initBackward() = 0;
	virtual int minNumInputs() = 0;
	virtual int minNumOutputs() = 0;	
	virtual void forward();
	virtual void backward();	
	std::string name() const;		
	virtual void traverse(NodeObserver *observer, TraverseOrder order, bool visit_condition);
	void setVisited(bool state);
	std::vector<std::shared_ptr<InputTerminal>> &inputs();
	std::vector<std::shared_ptr<OutputTerminal>> &outputs();
	std::shared_ptr<InputTerminal> input(int index);
	std::shared_ptr<OutputTerminal> output(int index);
	bool isInitialized() const;
	void setInitialized(bool status);
protected:	
	std::vector<std::shared_ptr<InputTerminal>> _inputs;
	std::vector<std::shared_ptr<OutputTerminal>> _outputs;
	std::string _name;	
	bool _visited = false;
	bool _initialized = false;	
	NodeParam _param;	
};

