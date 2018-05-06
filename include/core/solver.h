#pragma once

#include "nodes/variable.h"
#include "core/terminal.h"
#include "proto/deepflow.pb.h"

class Variable;

class DeepFlowDllExport Solver : public CudaHelper {
public:
	Solver(deepflow::SolverParam *param);
	virtual void apply(std::shared_ptr<Variable> var) = 0;
	virtual void init(std::shared_ptr<Variable> var) = 0;
	virtual std::string to_cpp() const = 0;
	deepflow::SolverParam *param() const;
	const std::string name() const;
	const std::string scope() const;
	bool has_the_same_param(std::shared_ptr<Solver> another) const;
	void set_learning_rate(float lr);
	void set_enabled(bool state);
protected:
	deepflow::SolverParam *_param;
	bool _initialized = false;
	float _learning_rate = 0.0f;	
	bool _enabled = true;
};