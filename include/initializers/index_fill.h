#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport IndexFill : public Initializer {
public:
	IndexFill(const InitParam &param);
	void apply(Variable *variable);
	void init() {}
};