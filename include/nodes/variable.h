#pragma once

#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class Initializer;

class DeepFlowDllExport Variable : public Node {
public:		
	Variable(std::shared_ptr<Initializer> initializer, const deepflow::NodeParam &param);
	virtual void initForward();
	virtual void initBackward();		
	virtual int minNumInputs() { return 0;  }
	virtual int minNumOutputs() { return 1; }	
	void transferDataToParam();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return ALWAYS_BACKWARD; }
	virtual std::string to_cpp() const;
protected:		
	std::shared_ptr<Initializer> _initializer;	
};
