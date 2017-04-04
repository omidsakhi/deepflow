#pragma once

#include "core/block.h"
#include "core/variable.h"

#include "proto/deepflow.pb.h"

class DeepFlowDllExport Solver : public CudaHelper {
public:
	Solver(std::shared_ptr<OutputTerminal> loss, SolverParam param);
	virtual void train_step() = 0;
	virtual void init() = 0;
protected:
	int _current_iteration = 0;
	std::shared_ptr<OutputTerminal> _loss_terminal;
	std::shared_ptr<Node> _loss_node;
	std::list<std::shared_ptr<Variable>> _variables;
	SolverParam _param;
	bool _initialized = false;
};