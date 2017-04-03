#pragma once

#include "core/node.h"

class DeepFlowDllExport Convolution2D : public Node {
public:
	Convolution2D(NodeParam param);
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
	cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo;
	size_t _fwdWorkspaceSize;
	size_t _bwdDataWorkspaceSize;
	size_t _bwdFilterWorkspaceSize;
	size_t _maxWorkspaceSize;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	float *d_workspace;

};
