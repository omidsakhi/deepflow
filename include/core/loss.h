#pragma once

#include "core/node.h"

class DeepFlowDllExport Loss : public Node {
public:
	Loss(const deepflow::NodeParam &_block_param);
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return ALWAYS_BACKWARD; }
};
