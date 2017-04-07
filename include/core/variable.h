#pragma once

#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class Initializer;

class DeepFlowDllExport Variable : public Node {
public:		
	Variable(std::shared_ptr<Initializer> initializer, const NodeParam &param);
	void initForward();
	void initBackward();	
	void forward();
	int minNumInputs() { return 0;  }
	int minNumOutputs() { return 1; }	
	void toImage(int iteration);
	bool snapshot();
	int snapshotInterval();
	void transferDataToParam();
protected:		
	std::shared_ptr<Initializer> _initializer;	
};
