#pragma once

#include "nodes/variable.h"

class Initializer;

class DeepFlowDllExport DataGenerator : public Variable {
public:
	DataGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "data_generator"; }
	bool is_generator() override;
	void forward() override;	
	std::string to_cpp() const override;
private:	
	int _batch_size = 0;
	bool _no_solver = false;	
};
