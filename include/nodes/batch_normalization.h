#pragma once

#include "core/node.h"

class DeepFlowDllExport BatchNormalization : public Node {
public:
	BatchNormalization(const deepflow::NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnHandle_t _cudnnHandle;
	cudnnBatchNormMode_t _batchNormMode;
	cudnnTensorDescriptor_t _xDesc, _yDesc, _bnScaleBiasMeanVarDesc;
	double _eps;
	double _exp_avg_factor = 0;
	float * _resultSaveMean;
	float * _resultSaveInvVariance;
	float *_resultRunningMean = nullptr;
	float *_resultRunningVariance = nullptr;
	float *_bnScale;
	float *_bnBias;
	float *_x;
	float *_y;
	float *_dy;
	float *_dx;
	size_t _size;
};