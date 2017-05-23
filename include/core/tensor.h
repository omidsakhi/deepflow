#pragma once


#include "core/export.h"
#include "core/cuda_helper.h"

#include <string>
#include <vector>
#include <array>

#include "proto/deepflow.pb.h"
#include "core/common_cu.h"

class DeepFlowDllExport Tensor : public CudaHelper {
public:
	
	enum TensorType
	{
		Float = 0,
		Double = 1,
		Half = 2,
		Int8 = 3,
		Int32 = 4,
		Int8x4 = 5
	};

	Tensor();
	Tensor(const deepflow::TensorParam &param);
	//Tensor(std::initializer_list<int> dims, TensorType type);
	Tensor(std::array<int, 4> dims, TensorType type);	
	cudnnDataType_t cudnnType() const;
	TensorType type() const;
	std::string typeString() const;
	void init();
	std::string shape() const;
	int size() const;
	size_t sizeInBytes() const;
	cudnnTensorDescriptor_t descriptor() const;
	int dim(int i) const;
	std::array<int, 4> dims() const;
	const void * data() const;
	void * mutableData();
	void reset();
	void release();
	
	template <typename T>
	int isValid() const;

	template <typename T>
	void print() const;

	template <typename T>
	std::shared_ptr<std::vector<T>> cpyToHost() const;

	float toFloat() const;

	std::string toString() const;
protected:
	std::array<int, 4> _dims;
	size_t _size;
	size_t _sizeInBytes;
	cudnnTensorDescriptor_t _desc = 0;
	TensorType _reader_type;
	std::string _shapeString;
	void *d_data;
};


template <typename T>
void Tensor::print() const {
	std::cout << "Shape: " << shape() << std::endl;
	auto h_data = cpyToHost<T>();
	for (int i = 0; i < _size; ++i) {
		std::cout << h_data->data()[i] << " ";
	}
	std::cout << std::endl;
}

template <typename T>
std::shared_ptr<std::vector<T>> Tensor::cpyToHost() const {
	auto h_data = std::make_shared<std::vector<T>>(_size);		
	cudaMemcpy(h_data->data(), d_data, _sizeInBytes, cudaMemcpyDeviceToHost);
	return h_data;
}
