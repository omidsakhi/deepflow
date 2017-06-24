#pragma once

#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class Initializer;

class DeepFlowDllExport Variable : public Node {
public:		
	Variable(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	virtual void initForward();
	virtual void initBackward();		
	virtual int minNumInputs() { return 0;  }
	virtual int minNumOutputs() { return 1; }
	virtual void forward();
	virtual void backward();
	void prep_for_saving();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType();
	virtual std::string to_cpp() const;
protected:		
	std::shared_ptr<Initializer> _initializer;	
};
