#pragma once

#include "core/export.h"
#include "core/initializer.h"
#include "proto/deepflow.pb.h"

#include <random>

class DeepFlowDllExport TruncatedNormal : public Initializer {
public:
	TruncatedNormal(deepflow::InitParam *param);
	void init() {}
	void apply(Variable *variable);
	std::string to_cpp() const;
private:
	std::random_device rd;
	std::mt19937 generator;
};