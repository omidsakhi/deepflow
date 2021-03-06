#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport GradientFill : public Initializer {
public:
	GradientFill(deepflow::InitParam *param);
	void apply(Node *node);
	void init() {}
	std::string to_cpp() const;
};