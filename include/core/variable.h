#pragma once

#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class Initializer;

class DeepFlowDllExport Variable : public Node {
public:		
	Variable(std::shared_ptr<Initializer> initializer, const NodeParam &param);	
	virtual void initForward();
	virtual void initBackward();		
	virtual int minNumInputs() { return 0;  }
	virtual int minNumOutputs() { return 1; }	
	void transferDataToParam();
protected:		
	std::shared_ptr<Initializer> _initializer;	
};
