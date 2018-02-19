#pragma once


#include "core/export.h"
#include "core/node.h"

#include "proto/deepflow.pb.h"

class DeepFlowDllExport PlaceHolder : public Node {
public:
	PlaceHolder(deepflow::NodeParam *param);
	void init();	
	void forward();
	void backward();	
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	std::string to_cpp() const;
};
