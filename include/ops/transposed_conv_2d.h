#pragma once

#include "core/node.h"

class DeepFlowDllExport TransposedConvolution2D : public Node {
public:
	TransposedConvolution2D(const NodeParam &param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
protected:
	cudnnHandle_t _cudnnHandle;
	cudnnFilterDescriptor_t _filterDesc;
	cudnnConvolutionDescriptor_t _convDesc;
	cudnnConvolutionFwdAlgo_t _fwdAlgo;	
	cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo;
	size_t _fwdWorkspaceSize;
	size_t _bwdDataWorkspaceSize;	
	size_t _maxWorkspaceSize;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	float *d_workspace;

};