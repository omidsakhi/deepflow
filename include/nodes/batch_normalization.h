#pragma once

#include "core/node.h"

class DeepFlowDllExport BatchNormalization : public Node {
public:
	BatchNormalization(deepflow::NodeParam *param);
	int minNumInputs() { return 3; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "batch_normalization"; }
	void init();	
	void forward();
	void backward();
	void prep_for_saving();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;
	cudnnBatchNormMode_t _bnMode;
	cudnnTensorDescriptor_t _xDesc, _yDesc, _bnScaleBiasMeanVarDesc;
	double _eps = CUDNN_BN_MIN_EPSILON;
	double _exp_avg_factor = 0.001f;
	float * _cachedMean = nullptr;
	float * _cachedVariance = nullptr;
	float * _runningMean = nullptr;
	float * _runningVariance = nullptr;
	size_t _bnScaleBiasMeanVarSize;
	size_t _bnScaleBiasMeanVarSizeInBytes;
};