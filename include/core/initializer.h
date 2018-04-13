#pragma once

#include "core/export.h"
#include "core/cuda_helper.h"
#include "core/tensor.h"
#include "proto/deepflow.pb.h"

class Node;

class DeepFlowDllExport Initializer : public CudaHelper {
public:
	Initializer(deepflow::InitParam *param);
	//Initializer(std::array<int, 4> dims, Tensor::TensorType type);
	virtual void init() = 0;
	virtual void apply(Node *node) = 0;	
	std::array<int, 4> dims() const;	
	deepflow::InitParam *param();
	virtual std::string to_cpp() const = 0;
protected:	
	deepflow::InitParam *_param;
	std::array<int, 4> _dims;
	Tensor::TensorType _reader_type;
};

