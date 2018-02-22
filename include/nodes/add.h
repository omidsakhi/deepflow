#pragma once

#include "core/node.h"

class DeepFlowDllExport Add : public Node {
public:
	Add(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }	
	void init();		
	void forward();
	void backward();
	void setAlpha(float alpha);
	void setBeta(float beta);
	std::string to_cpp() const;
private:
	float _alpha, _beta;
};
