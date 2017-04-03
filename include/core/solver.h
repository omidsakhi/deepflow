#pragma once

#include "core/block.h"

#include "proto/deepflow.pb.h"

class DeepFlowDllExport Solver {
public:
	Solver(std::shared_ptr<OutputTerminal> loss, SolverParam param);
	virtual void step() = 0;
	virtual void train() = 0;
protected:
	int _current_iteration;
	std::shared_ptr<OutputTerminal> _loss_terminal;
	std::shared_ptr<Node> _loss_node;
	std::list< Node* > _variables;
	SolverParam _param;
};