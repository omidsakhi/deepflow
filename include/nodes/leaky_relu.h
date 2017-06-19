#pragma once

#include "core/node.h"

#include <random>

class DeepFlowDllExport LeakyRelu : public Node {
public:
	LeakyRelu(deepflow::NodeParam *param);
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
	float _initial_negative_slope = 0;
	float _negative_slope = 0;
	bool _randomize = false;
	std::random_device _random_device;
};
