#pragma once

#include "nodes/variable.h"
#include "core/terminal.h"
#include "proto/deepflow.pb.h"

class Variable;

class DeepFlowDllExport Solver : public CudaHelper {
public:
	Solver(const deepflow::SolverParam &param);
	virtual void apply(std::shared_ptr<Variable> var) = 0;
	virtual void init(std::shared_ptr<Variable> var) = 0;
	virtual std::string to_cpp() const = 0;
	const deepflow::SolverParam& param() const;
	const std::string name() const;
	bool hasTheSameParam(std::shared_ptr<Solver> another) const;
	NodeInputPtr enable_input();
	void create_enable_input();
protected:
	NodeInputPtr _enable_input;
	deepflow::SolverParam _param;
	bool _initialized = false;
};