#pragma once

#include "core/export.h"
#include "core/initializer.h"
#include "proto/deepflow.pb.h"

class DeepFlowDllExport RandomUniform : public Initializer {
public:
	RandomUniform(const deepflow::InitParam &_block_param);
	void init() {}
	void apply(Variable *variable);
	std::string to_cpp() const;
};