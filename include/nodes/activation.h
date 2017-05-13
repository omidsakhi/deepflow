#pragma once


#include "core/node.h"

class DeepFlowDllExport Activation : public Node {
public:
	Activation(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnActivationDescriptor_t _activation_desc;
	cudnnHandle_t _cudnnHandle;
	const float alpha = 1.0f;
	const float beta = 0.0f;
};
