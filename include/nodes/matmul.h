#pragma once

#include "core/node.h"

#include <cublas_v2.h>

class DeepFlowDllExport MatMul : public Node {
public:
	MatMul(const deepflow::NodeParam &_block_param);
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
	float _alpha;
	float _beta;
	cublasHandle_t _handle;
	int _col_A, _row_A, _col_B, _row_B;	
};