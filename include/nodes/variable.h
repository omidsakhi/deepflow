#pragma once

#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class Initializer;

class DeepFlowDllExport Variable : public Node {
public:		
	Variable(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	virtual void init();	
	virtual int minNumInputs() { return 0;  }
	virtual int minNumOutputs() { return 1; }
	virtual void forward();
	virtual void backward();
	float * gradients();
	void reset_gradients();
	void prep_for_saving();
	void clamp(float min, float max);
	virtual std::string to_cpp() const;
protected:		
	std::shared_ptr<Initializer> _initializer;
	float * _grad = nullptr;
};
