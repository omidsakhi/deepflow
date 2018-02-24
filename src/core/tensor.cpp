#include "core/tensor.h"

Tensor::Tensor() {
	_size = _sizeInBytes = 0;
	d_data = NULL;
}

Tensor::Tensor(deepflow::TensorParam *param) {
	switch (param->dims_size()) {
	case 1:
		_dims = { 1,param->dims(0),1,1 };
		break;
	case 2:
		_dims = { param->dims(0),param->dims(1),1,1 };
		break;
	case 3:
		_dims = { param->dims(0), 1, param->dims(1),param->dims(2) };
		break;
	case 4:
		_dims = { param->dims(0),param->dims(1),param->dims(2),param->dims(3) };
		break;
	default:
		LOG(FATAL) << "Unsupported shape.";
	}
	_reader_type = (Tensor::TensorType) param->type();
	init();
}

Tensor::Tensor(std::array<int, 4> dims, TensorType type) {
	_dims = dims;
	_reader_type = type;
	init();
}

Tensor::Tensor(std::array<int, 4> dims, std::shared_ptr<Tensor> shadow_tensor)
{
	_dims = dims;
	_reader_type = shadow_tensor->type();
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];
	_shapeString = std::to_string(_dims[0]);
	for (int i = 1; i < 4; ++i)
		_shapeString += "x" + std::to_string(_dims[i]);
	LOG_IF(FATAL, _size != shadow_tensor->size()) << "The tensor that you are shadowing must have the same size.";
	DF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_desc));
	DF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, (cudnnDataType_t)_reader_type, _dims[0], _dims[1], _dims[2], _dims[3]));
	DF_CUDNN_CHECK(cudnnGetTensorSizeInBytes(_desc, &_sizeInBytes));
	d_data = shadow_tensor->mutableData();
}

void Tensor::init() {
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];	
	_shapeString = std::to_string(_dims[0]);
	for (int i = 1; i < 4; ++i)
		_shapeString += "x" + std::to_string(_dims[i]);	
	DF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_desc));
	DF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, (cudnnDataType_t)_reader_type, _dims[0], _dims[1], _dims[2], _dims[3]));
	DF_CUDNN_CHECK(cudnnGetTensorSizeInBytes(_desc, &_sizeInBytes));	
	DF_CUDA_CHECK(cudaMalloc(&d_data, _sizeInBytes));	
}

cudnnDataType_t Tensor::cudnnType() const {
	return (cudnnDataType_t)_reader_type;
}

Tensor::TensorType Tensor::type() const {
	return _reader_type;
}

std::string Tensor::shape() const {
	return _shapeString;
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

int Tensor::isValid() const {
	switch (_reader_type) {	
	case TensorType::Int8:
	{
		auto h_data = cpyToHost<unsigned char>();
		for (int i = 0; i < h_data->size(); ++i) {
			if (isinf((float)h_data->at(i)))
				return -1;
			else if (isnan((float)h_data->at(i)))
				return -2;
		}		
	}
	break;
	case TensorType::Int32:
	{
		auto h_data = cpyToHost<int>();
		for (int i = 0; i < h_data->size(); ++i) {
			if (isinf((float)h_data->at(i)))
				return -1;
			else if (isnan((float)h_data->at(i)))
				return -2;
		}
	}
	break;
	case TensorType::Float:
	{
		auto h_data = cpyToHost<float>();
		for (int i = 0; i < h_data->size(); ++i) {
			if (isinf(h_data->at(i)))
				return -1;
			else if (isnan(h_data->at(i)))
				return -2;
		}
	}
	break;
	default:
		LOG(FATAL) << "Unsupported type for toString() function";
	};
	return 0;
}



void Tensor::release() {
	if (d_data)
	{
		DF_CUDA_CHECK(cudaFree(d_data));		
		d_data = 0;
		_size = 0;
		_sizeInBytes = 0;		
	}
}

void Tensor::reset() {
	DF_CUDA_CHECK(cudaMemset(d_data, 0, _sizeInBytes));	
}

float Tensor::toFloat() const {
	LOG_IF(FATAL, _reader_type != TensorType::Float) << "_type != TensorType::Float";
	LOG_IF(FATAL, _size != 1) << "toFloat() - _size != 1 [FAILED]";
	auto h_data = cpyToHost<float>();
	return h_data->at(0);
}

std::string Tensor::toString() const {
	std::string output;
	switch (_reader_type) {
		case TensorType::Int8:
		{
			auto h_data = cpyToHost<unsigned char>();						
			for (int i = 0; i < h_data->size(); ++i)
				output += std::to_string(h_data->at(i)) + " ";
		}
		break;
		case TensorType::Int32:
		{
			auto h_data = cpyToHost<int>();
			for (int i = 0; i < h_data->size(); ++i)
				output += std::to_string(h_data->at(i)) + " ";
		}
		break;
		case TensorType::Float:
		{
			auto h_data = cpyToHost<float>();
			for (int i = 0; i < h_data->size(); ++i)
				output += std::to_string(h_data->at(i)) + " ";
		}
		break;
		default:
			LOG(FATAL) << "Unsupported type for toString() function";
	};
	return output;
}

std::string Tensor::typeString() const {
	switch (_reader_type) {
	case Float:
		return "Float";
		break;
	case Double:
		return "Double";
		break;
	case Half:
		return "Half";
		break;
	case Int8:
		return "Int8";
		break;
	case Int32:
		return "Int32";
		break;
	case Int8x4:
		return "Int8x4";
		break;
	default:
		LOG(FATAL) << "Unsupported type";
	};
	return "";
}