#pragma once


#include "core/node.h"

class DeepFlowDllExport Psnr : public Node {
public:
	enum PrintTime {
		EVERY_PASS = 0,
		END_OF_EPOCH = 1
	};
	Psnr(const NodeParam &param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	PrintTime _print_time;
	float *d_square_error;
	float *d_sum_square_error;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cudnnHandle_t _cudnnHandle;	
	cudnnReduceTensorDescriptor_t _reduce_tensor_desciptor;
	cudnnTensorDescriptor_t _output_desc;
	float *_d_workspace;
	size_t _workspaceSizeInBytes;
};
