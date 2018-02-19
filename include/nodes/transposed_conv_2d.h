#pragma once

#include "core/node.h"

class DeepFlowDllExport TransposedConvolution2D : public Node {
public:
	TransposedConvolution2D(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
private:
	cudnnHandle_t _cudnnHandle;
	cudnnFilterDescriptor_t _wDesc;
	cudnnConvolutionDescriptor_t _convDesc;
	cudnnConvolutionFwdAlgo_t _fwdAlgo;	
	cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo;
	size_t _fwdWorkspaceSize = 0;
	size_t _bwdDataWorkspaceSize = 0;	
	size_t _maxWorkspaceSize = 0;
	size_t _bwdFilterWorkspaceSize = 0;
	float *d_workspace;
	float *_dy;

};