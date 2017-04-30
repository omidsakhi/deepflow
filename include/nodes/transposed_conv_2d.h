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
	std::string to_cpp() const;
	ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
private:
	cudnnHandle_t _cudnnHandle;
	cudnnFilterDescriptor_t _filterDesc;
	cudnnConvolutionDescriptor_t _convDesc;
	cudnnConvolutionFwdAlgo_t _fwdAlgo;	
	cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo;
	size_t _fwdWorkspaceSize;
	size_t _bwdDataWorkspaceSize;	
	size_t _maxWorkspaceSize;
	size_t _bwdFilterWorkspaceSize;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	float *d_workspace;

};