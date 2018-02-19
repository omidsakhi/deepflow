#pragma once

#include "core/node.h"

class DeepFlowDllExport BatchNormalization : public Node {
public:
	BatchNormalization(deepflow::NodeParam *param);
	int minNumInputs() { return 3; }
	int minNumOutputs() { return 1; }	
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
	float * _resultSaveMean = nullptr;
	float * _resultSaveInvVariance = nullptr;
	float *_bnScale = nullptr;
	float *_bnBias = nullptr;
	float *_resultBnScaleDiff = nullptr;
	float *_resultBnBiasDiff = nullptr;
	float *_x = nullptr;
	float *_y = nullptr;
	float *_dy = nullptr;
	float *_dx = nullptr;
	size_t _bnScaleBiasMeanVarSize;
	size_t _bnScaleBiasMeanVarSizeInBytes;
};