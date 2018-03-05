#include "core/tensor.h"

#include <vector>

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
	_name = param->name();
	init();
}

Tensor::Tensor(std::array<int, 4> dims, TensorType type, std::string name, bool reset) {
	_dims = dims;
	_reader_type = type;
	_name = name;
	init(reset);
}

Tensor::Tensor(std::array<int, 4> dims, std::shared_ptr<Tensor> shadow_tensor, std::string name)
{
	_dims = dims;
	_reader_type = shadow_tensor->type();
	_name = name;
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

void Tensor::init(bool reset_to_zero) {
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];	
	_shapeString = std::to_string(_dims[0]);
	for (int i = 1; i < 4; ++i)
		_shapeString += "x" + std::to_string(_dims[i]);	
	DF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_desc));
	DF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, (cudnnDataType_t)_reader_type, _dims[0], _dims[1], _dims[2], _dims[3]));
	DF_CUDNN_CHECK(cudnnGetTensorSizeInBytes(_desc, &_sizeInBytes));	
	DF_CUDA_CHECK(cudaMalloc(&d_data, _sizeInBytes));
	if (reset_to_zero)
		reset();
		
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

void Tensor::statistics(double * mean, double * std, double * min, double * max) const
{
	double eps = 10e-7;
	auto h_data = cpyToHost<float>();
	double sum = 0;
	int count = 0;
	*min = DBL_MAX;
	*max = -DBL_MAX;
	for (int i = 0; i < _size; ++i) {
		float val = h_data->at(i);
		sum += val;
		if (*max < val)
			*max = val;
		if (*min > val)
			*min = val;
		count++;
	}
	*mean = sum / count;
	sum = 0;
	for (int i = 0; i < _size; ++i) {
		float val = h_data->at(i);		
		sum += (val - *mean) * (val - *mean);
	}
	*std = sqrt( sum /(count - 1));
	if (fabs(*mean) < eps)
		*mean = 0;
	if (fabs(*max) < eps)
		*max = 0;
	if (fabs(*min) < eps)
		*min = 0;
	if (fabs(*std) < eps)
		*std = 0;
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

void Tensor::set(std::initializer_list<float> values)
{
	LOG_IF(FATAL, _reader_type != TensorType::Float) << "_type != TensorType::Float";
	LOG_IF(FATAL, values.size() != size()) << "values.size() != size()";
	std::vector<float> h_data = values;	
	cpyFromHost<float>(h_data);
}

bool Tensor::verify(std::initializer_list<float> values)
{
	auto h_data = cpyToHost<float>();
	bool res = true;
	int index = 0;
	for (auto v : values) {
		if (h_data->at(index++) != v) {
			res = false;
			break;
		}
	}
	return res;
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

std::string Tensor::name() const
{
	return _name;
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