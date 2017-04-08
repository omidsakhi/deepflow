#pragma once

#include "core/block.h"
#include "core/variable.h"

#include "proto/deepflow.pb.h"

class DeepFlowDllExport Solver : public CudaHelper {
public:
	Solver(NodeOutputPtr loss, const SolverParam &param);
	virtual void train_step() = 0;
	virtual void init() = 0;
	const SolverParam& param() const;
	int maxIteration();
protected:
	int _current_step = 0;
	NodeOutputPtr _loss_terminal;
	std::shared_ptr<Node> _loss_node;
	std::list<std::shared_ptr<Variable>> _variables;
	SolverParam _param;
	bool _initialized = false;
};