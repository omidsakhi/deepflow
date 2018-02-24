#pragma once

#include "nodes/variable.h"

class Initializer;

class DeepFlowDllExport DataGenerator : public Variable {
public:
	DataGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "data_generator"; }
	bool isGenerator();
	void forward();
	bool isLastBatch();
	std::string to_cpp() const;
private:
	int _current_batch = 0;
	int _num_total_samples = 0;
	int _num_batches = 0;
	int _batch_size = 0;
	bool _no_solver = false;
	bool _last_batch = false;
};
