#pragma once

#include "core/node.h"

class DeepFlowDllExport Convolution2D : public Node {
public:
	Convolution2D(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }	
	std::string op_name() const override { return "conv2d"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
protected:
	cudnnHandle_t _cudnnHandle;
	void *_x, *_y, *_w, *_dx, *_dy, *_dw;	
	cudnnTensorDescriptor_t _xDesc, _yDesc, _dxDesc, _dyDesc;
	cudnnFilterDescriptor_t _wDesc;	
	cudnnConvolutionDescriptor_t _convDesc;
	cudnnConvolutionFwdAlgo_t _fwdAlgo;
	cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo;
	cudnnActivationDescriptor_t _activationDesc;
	size_t _fwdWorkspaceSize;
	size_t _bwdDataWorkspaceSize;
	size_t _bwdFilterWorkspaceSize;
	size_t _maxWorkspaceSize;
	float *d_workspace;	
};
