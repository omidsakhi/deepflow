#pragma once

#include "core/export.h"
#include "core/node.h"

class PlaceHolder;

class DeepFlowDllExport Reader : public Node {
public:	
	Reader(NodeParam param);
	virtual void nextBatch() = 0;
	virtual bool isLastBatch() = 0;
};

