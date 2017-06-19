#pragma once

#include "core/node.h"

class DeepFlowDllExport Convolution2D : public Node {
public:
	Convolution2D(deepflow::NodeParam *param);
	int minNumInputs();
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return DEPENDS_ON_OUTPUTS; }
	virtual BackwardType backwardType() { return DEPENDS_ON_INPUTS; }
	std::string to_cpp() const;
protected:
	cudnnHandle_t _cudnnHandle;
	void *_x, *_y, *_w, *_dx, *_dy, *_dw, *_b, *_db, *_z, *_dz;	
	cudnnTensorDescriptor_t _xDesc, _yDesc, _dxDesc, _dyDesc, _zDesc, _dzDesc, _bDesc, _dbDesc;
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
	int _num_inputs;
};
