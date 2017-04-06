#include "core/tensor.h"

Tensor::Tensor() {
	_size = _sizeInBytes = 0;
	d_data = NULL;
}

Tensor::Tensor(const TensorParam &param) {
	switch (param.dims_size()) {
	case 1:
		_dims = { 1,param.dims(0),1,1 };
		break;
	case 2:
		_dims = { param.dims(0),param.dims(1),1,1 };
		break;
	case 3:
		_dims = { param.dims(0), 1, param.dims(1),param.dims(2) };
		break;
	case 4:
		_dims = { param.dims(0),param.dims(1),param.dims(2),param.dims(3) };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}
	_type = (Tensor::TensorType) param.type();
	init();
}

/*
Tensor::Tensor(std::initializer_list<int> dims, TensorType type) {
	std::vector<int> values(dims);
	switch (values.size()) {
	case 1:
		_dims = { 1, values[0], 1, 1 };
		break;
	case 2:
		_dims = { values[0], values[1], 1, 1 };
		break;
	case 3:
		_dims = { values[0], 1, values[1], values[2] };
		break;
	case 4:
		_dims = { values[0], values[1], values[2], values[3] };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}
	_type = type;
	init();
}
*/

Tensor::Tensor(std::array<int, 4> dims, TensorType type) {
	_dims = dims;
	_type = type;
	init();
}

void Tensor::init() {
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];	
	_string = std::to_string(_dims[0]);
	for (int i = 1; i < 4; ++i)
		_string += "x" + std::to_string(_dims[i]);
	LOG_IF(FATAL, cudnnCreateTensorDescriptor(&_desc) != 0) << "cudnnCreateTensorDescriptor [FAILED]";
	LOG_IF(FATAL, cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, (cudnnDataType_t) _type, _dims[0], _dims[1], _dims[2], _dims[3]) != 0) << "cudnnSetTensor4dDescriptor [FAILED] " << _string;
	LOG_IF(FATAL, cudnnGetTensorSizeInBytes(_desc, &_sizeInBytes) != 0) << "cudnnGetTensorSizeInBytes [FAILED]";
	LOG_IF(FATAL, cudaMalloc(&d_data, _sizeInBytes) != 0) << "cudaMalloc [FAILED]";
}

cudnnDataType_t Tensor::cudnnType() const {
	return (cudnnDataType_t)_type;
}

Tensor::TensorType Tensor::type() const {
	return _type;
}

std::string Tensor::shape() const {
	return _string;
}

int Tensor::size() const {
	return _size;
}

size_t Tensor::sizeInBytes() const {
	return _sizeInBytes;
}

cudnnTensorDescriptor_t Tensor::descriptor() const {
	return _desc;
}

int Tensor::dim(int i) const {
	return _dims[i];
}

std::array<int, 4> Tensor::dims() const {
	return _dims;
}

const void * Tensor::data() const {
	return d_data;
}

void * Tensor::mutableData() {
	return d_data;
}

template <typename T>
int Tensor::isValid() const {
	auto h_data = cpyToHost();
	for (int i = 0; i < _size; ++i) {
		if (isinf(h_data[i]))
			return -1;
		else if (isnan(h_data[i]))
			return -2;
	}
	return 0;
}



void Tensor::release() {
	if (d_data)
	{
		LOG_IF(FATAL, cudaFree(d_data) != 0);
		d_data = 0;
		_size = 0;
		_sizeInBytes = 0;		
	}
}

void Tensor::reset() {
	LOG_IF(FATAL, cudaMemset(d_data, 0, _sizeInBytes) != 0) << "cudaMemset [FAILED]";
}

float Tensor::toFloat() const {
	LOG_IF(FATAL, _type != TensorType::Float) << "_type != TensorType::Float";
	LOG_IF(FATAL, _size != 1) << "_size != 1";
	auto h_data = cpyToHost<float>();
	return h_data->at(0);
}