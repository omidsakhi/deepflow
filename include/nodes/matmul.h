#pragma once

#include "core/node.h"

#include <cublas_v2.h>

class DeepFlowDllExport MatMul : public Node {
public:
	MatMul(deepflow::NodeParam *param);
	int minNumInputs() { return 2; }
	int minNumOutputs() { return 1; }
	void init();		
	void forward();
	void backward();
	std::string to_cpp() const;
private:	
	cublasHandle_t _handle;
	int _col_A, _row_A, _col_B, _row_B;	
};