#include "core/tensor.h"

#include <vector>

#include <mutex>

size_t Tensor::_used_gpu_mem_size = 0;

Tensor::Tensor() {}

Tensor::Tensor(std::array<int, 4> dims, std::string name, DataPolicy policy) {
	_dims = dims;	
	_name = name;
	init(policy);
}

Tensor::Tensor(std::array<int, 4> dims, std::shared_ptr<Tensor> shadow_tensor, std::string name)
{	
	_dims = dims;	
	_name = name;
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];
	_shapeString = std::to_string(_dims[0]);
	for (int i = 1; i < 4; ++i)
		_shapeString += "x" + std::to_string(_dims[i]);
	LOG_IF(FATAL, _size != shadow_tensor->size()) << "The tensor that you are shadowing must have the same size.";
	DF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_desc));
	DF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _dims[0], _dims[1], _dims[2], _dims[3]));
	DF_CUDNN_CHECK(cudnnGetTensorSizeInBytes(_desc, &_bytes));
	_shadow_tensor = shadow_tensor;
	_location = SHADOW;
	cudaStreamCreate(&_stream);	
}

void Tensor::init(DataPolicy policy) {
	_size = _dims[0] * _dims[1] * _dims[2] * _dims[3];	
	_shapeString = std::to_string(_dims[0]);
	_policy = policy;
	for (int i = 1; i < 4; ++i)
		_shapeString += "x" + std::to_string(_dims[i]);	
	DF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_desc));
	DF_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _dims[0], _dims[1], _dims[2], _dims[3]));
	DF_CUDNN_CHECK(cudnnGetTensorSizeInBytes(_desc, &_bytes));
	if (_policy == GPU_ONLY_POLICY) {
		_cpu_data = nullptr;
		DF_CUDA_CHECK(cudaMalloc(&_gpu_data, _bytes));
		_used_gpu_mem_size += _bytes;
		DF_CUDA_CHECK(cudaMemset(_gpu_data, 0, _bytes));
		_location = GPU;
	}
	else if (_policy == GPU_WITH_CPU_OFFLOAD_POLICY) {
		_gpu_data = nullptr;
		try
		{
			_cpu_data = new float[_size];
		}
		catch (std::bad_alloc&)
		{
			LOG(FATAL) << "[FAILED] - host memory alocation failed for " << _name;
		}
		LOG_IF(FATAL, _cpu_data == nullptr);
		memset(_cpu_data, 0, _bytes);
		_location = CPU;
	}
	else if (_policy == CUDA_MANAGED_POLICY) {
		_cpu_data = nullptr;
		DF_CUDA_CHECK(cudaMallocManaged(&_gpu_data, _bytes));
		_used_gpu_mem_size += _bytes;
		DF_CUDA_CHECK(cudaMemset(_gpu_data, 0, _bytes));
		_location = CUDA_MANAGED;
	}
	cudaStreamCreate(&_stream);	
}

std::string Tensor::shape() const {
	return _shapeString;
}

int Tensor::size() const {
	return _size;
}

size_t Tensor::bytes() const {
	return _bytes;
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

void Tensor::offload_data()
{
	if (_offload_event) {
		cudaEventSynchronize(_offload_event);		
	}
	if (_location == SHADOW) {
		//LOG(INFO) << "Tensor " << _name << " CPU shahdow access from " << caller;
		_shadow_tensor->offload_data();
		return;
	}
	if (_location == GPU && _policy == GPU_WITH_CPU_OFFLOAD_POLICY) {
		LOG_IF(FATAL, _cpu_data == nullptr);
		LOG_IF(FATAL, _gpu_data == nullptr);
		if (_offload_event) {
			cudaEventDestroy(_offload_event);
			_offload_event = nullptr;
		}
		cudaEventCreate(&_offload_event);
		DF_CUDA_CHECK(cudaMemcpyAsync(_cpu_data, _gpu_data, _bytes, cudaMemcpyDeviceToHost, _stream));
		DF_CUDA_CHECK(cudaFree(_gpu_data));
		_used_gpu_mem_size -= _bytes;
		cudaEventRecord(_offload_event);
		_location = CPU;
		_gpu_data = nullptr;		
	}
}

float * Tensor::cpu_data() {
	if (_offload_event) {
		cudaEventSynchronize(_offload_event);
	}
	if (_location == SHADOW) {
		//LOG(INFO) << "Tensor " << _name << " CPU shahdow access from " << caller;
		return _shadow_tensor->cpu_data();
	}
	else if (_location == CUDA_MANAGED) {
		return _gpu_data;
	}
	else if (_location == GPU) {
		LOG_IF(FATAL, _policy == GPU_ONLY_POLICY);
		//LOG(INFO) << "Tensor " << _name << " switched to CPU from " << caller;
		LOG_IF(FATAL, _cpu_data == nullptr);
		LOG_IF(FATAL, _gpu_data == nullptr);
		DF_CUDA_CHECK(cudaMemcpy(_cpu_data, _gpu_data, _bytes, cudaMemcpyDeviceToHost));
		_location = CPU;
		DF_CUDA_CHECK(cudaFree(_gpu_data));
		_used_gpu_mem_size -= _bytes;
		_gpu_data = nullptr;		
		return _cpu_data;
	}
	else if (_location == CPU) {
		LOG_IF(FATAL, _policy == GPU_ONLY_POLICY);
		LOG_IF(FATAL, _cpu_data == nullptr);
		//LOG(INFO) << "Tensor " << _name << " CPU access from " << caller;
		return _cpu_data;
	}
	return nullptr;
}

float * Tensor::gpu_data() {
	if (_offload_event) {
		cudaEventSynchronize(_offload_event);
	}
	if (_location == SHADOW) {		
		//LOG(INFO) << "Tensor " << _name << " GPU shahdow access from " << caller;
		return _shadow_tensor->gpu_data();
	}
	else if (_location == GPU || _location == CUDA_MANAGED) {
		LOG_IF(FATAL, _gpu_data == nullptr);
		//LOG(INFO) << "Tensor " << _name << " GPU access from " << caller;
		return _gpu_data;
	}
	else if (_location == CPU) {
		//LOG(INFO) << "Tensor " << _name << " switched to GPU from " << caller;
		LOG_IF(FATAL, _policy == GPU_ONLY_POLICY);
		LOG_IF(FATAL, _cpu_data == nullptr);
		LOG_IF(FATAL, _gpu_data != nullptr);
		DF_CUDA_CHECK(cudaMalloc(&_gpu_data, _bytes));
		_used_gpu_mem_size += _bytes;
		LOG_IF(FATAL, _gpu_data == nullptr);
		DF_CUDA_CHECK(cudaMemcpy(_gpu_data, _cpu_data, _bytes, cudaMemcpyHostToDevice));
		_location = GPU;		
		return _gpu_data;
	}
	return nullptr;
}

int Tensor::is_valid() {	
	auto data = to_vec();
	for (int i = 0; i < _size; ++i) {
		if (isinf(data->at(i)))
			return -1;
		else if (isnan(data->at(i)))
			return -2;
	}
	return 0;
}

void Tensor::statistics(double * mean, double * std, double * min, double * max)
{	
	double eps = 10e-7;
	auto data = to_vec();
	double sum = 0;
	int count = 0;
	*min = DBL_MAX;
	*max = -DBL_MAX;
	for (int i = 0; i < _size; ++i) {
		float val = data->at(i);
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
		float val = data->at(i);
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

void Tensor::print()
{	
	std::cout << "Shape: " << shape() << std::endl;
	auto data = to_vec();
	for (int i = 0; i < _size; ++i) {
		std::cout << data->at(i) << " ";
	}
	std::cout << std::endl;

}


void Tensor::release() {		
	if (_location == SHADOW) {
		_shadow_tensor->release();
	}
	else if (_location == CPU) {
		LOG_IF(FATAL, _gpu_data != nullptr);
		LOG_IF(FATAL, _cpu_data == nullptr);
		free(_cpu_data);
		_cpu_data = nullptr;
	}
	else {
		LOG_IF(FATAL, _gpu_data == nullptr);
		if (_gpu_data) {
			cudaFree(_gpu_data);
			_used_gpu_mem_size -= _bytes;
			_gpu_data = nullptr;
		}
		if (_cpu_data) {
			free(_cpu_data);
			_cpu_data = nullptr;
		}
	}
}

void Tensor::reset() {	
	if (_location == SHADOW) {
		_shadow_tensor->reset();
	}
	else if (_location == CPU) {
		LOG_IF(FATAL, _cpu_data == nullptr);
		memset(_cpu_data, 0, _bytes);
	}
	else if (_location == GPU || _location == CUDA_MANAGED) {
		LOG_IF(FATAL, _gpu_data == nullptr);		
		DF_CUDA_CHECK(cudaMemset(_gpu_data, 0, _bytes));
	}	
}

void Tensor::set(const std::vector<float> &values)
{	
	LOG_IF(FATAL, values.size() != size()) << "values.size() != size()";
	if (_location == SHADOW) {
		_shadow_tensor->set(values);
	}
	else if (_location == CPU) {
		LOG_IF(FATAL, _cpu_data == nullptr);
		memcpy(_cpu_data, values.data(), _bytes);
	}
	else if (_location == GPU || _location == CUDA_MANAGED) {
		LOG_IF(FATAL, _gpu_data == nullptr);
		DF_CUDA_CHECK(cudaMemcpy(_gpu_data, values.data(), _bytes, cudaMemcpyHostToDevice));
	}
}

bool Tensor::verify(const std::vector<float> &values)
{		
	float eps = 10e-7;
	auto data = to_vec();
	bool res = true;
	int index = 0;
	for (auto v : values) {
		float value = data->at(index);
		if (fabs(value - v) > eps) {
			LOG(INFO) << value << " != " << v << " @ " << index;
			res = false;
			break;
		}
		index++;
	}
	return res;
}

float Tensor::to_float() {		
	LOG_IF(FATAL, _size != 1) << "toFloat() - _size != 1 [FAILED]";
	auto data = to_vec();	
	return data->at(0);
}

std::shared_ptr<std::vector<float>> Tensor::to_vec()
{
	if (_offload_event) {
		cudaEventSynchronize(_offload_event);
	}
	if (_location == SHADOW)
		return _shadow_tensor->to_vec();
	auto vec = std::make_shared<std::vector<float>>(_size);
	if (_location == GPU || _location == CUDA_MANAGED) {
		DF_CUDA_CHECK(
			cudaMemcpy(vec->data(), _gpu_data, _bytes, cudaMemcpyDeviceToHost)
		);
	}
	else {
		memcpy(vec->data(), _cpu_data, _bytes);
	}
	return vec;
}

std::string Tensor::toString() {	
	std::string output;
	auto data = to_vec();
	for (int i = 0; i < _size; ++i)
		output += std::to_string(data->at(i)) + " ";
	return output;
}

std::string Tensor::name() const
{
	return _name;
}

size_t Tensor::used_gpu()
{
	return _used_gpu_mem_size;	
}
