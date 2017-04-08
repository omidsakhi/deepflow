#pragma once


#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class DeepFlowDllExport PlaceHolder : public Node {
public:
	PlaceHolder(const NodeParam &param);	
	void initForward();
	void initBackward();
	void forward();
	void backward();
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
};
