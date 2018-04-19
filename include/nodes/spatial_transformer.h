#pragma once

#include "core/node.h"

class DeepFlowDllExport SpatialTransformer : public Node {
public:
	SpatialTransformer(deepflow::NodeParam *param);
	int minNumInputs() { return 3; } // input / theta / grid
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "spatial_transformer"; }
	void init() override;
	void forward() override;
	void backward() override;
	std::string to_cpp() const override;
private:
	cudnnHandle_t _cudnnHandle = nullptr;
	cudnnSpatialTransformerDescriptor_t _stDesc = nullptr;	
};

