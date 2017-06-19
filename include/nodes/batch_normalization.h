#pragma once

#include "core/node.h"

class DeepFlowDllExport BatchNormalization : public Node {
public:
	BatchNormalization(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	void prep_for_saving();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnHandle_t _cudnnHandle;
	cudnnBatchNormMode_t _bnMode;
	cudnnTensorDescriptor_t _xDesc, _yDesc, _bnScaleBiasMeanVarDesc;
	double _eps = 0;
	double _exp_avg_factor = 0;
	float * _bnMean = nullptr;
	float * _bnInvVariance = nullptr;
	float *_resultRunningMean = nullptr;
	float *_resultRunningVariance = nullptr;
	float *_bnScale = nullptr;
	float *_bnBias = nullptr;
	float *_x = nullptr;
	float *_y = nullptr;
	float *_dy = nullptr;
	float *_dx = nullptr;
	size_t _bnScaleBiasMeanVarSize;
	size_t _bnScaleBiasMeanVarSizeInBytes;
};