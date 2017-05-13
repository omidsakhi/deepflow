#pragma once

#include "core/export.h"
#include "core/node.h"

class PlaceHolder;

class DeepFlowDllExport Generator {
public:	
	Generator(const deepflow::NodeParam &_block_param);
	virtual void nextBatch() = 0;
	virtual bool isLastBatch() = 0;
};

