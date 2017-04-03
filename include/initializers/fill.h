#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport Fill : public Initializer {
public:
	Fill(InitParam param);
	void apply(Variable *variable);
	void init() {}
};