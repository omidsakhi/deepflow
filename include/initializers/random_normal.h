#pragma once

#include "core/export.h"
#include "core/initializer.h"
#include "proto/deepflow.pb.h"

class DeepFlowDllExport RandomNormal : public Initializer {
public:
	RandomNormal(const deepflow::InitParam &_block_param);
	void init() {}
	void apply(Variable *variable);
	std::string to_cpp() const;
};