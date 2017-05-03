#pragma once

#include "core/export.h"
#include "core/initializer.h"

class DeepFlowDllExport Fill : public Initializer {
public:
	Fill(const deepflow::InitParam &param);
	void apply(Variable *variable);
	void init() {}
	std::string to_cpp() const;
};