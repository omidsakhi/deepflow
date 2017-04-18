#pragma once

#include "core/export.h"
#include "core/node.h"

class PlaceHolder;

class DeepFlowDllExport Generator {
public:	
	Generator(const NodeParam &param);
	virtual void nextBatch() = 0;
	virtual bool isLastBatch() = 0;
};

