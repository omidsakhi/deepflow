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
	
	enum DataLocation
	{
		CPU,
		GPU,
		CUDA_MANAGED,
		SHADOW
	};

	enum DataPolicy
	{
		GPU_ONLY_POLICY,
		GPU_WITH_CPU_OFFLOAD_POLICY,
		CUDA_MANAGED_POLICY
	};

	Tensor();	
	Tensor(std::array<int, 4> dims, std::string name, DataPolicy policy);
	Tensor(std::array<int, 4> dims, std::shared_ptr<Tensor> shadow_tensor, std::string name);
	void init(DataPolicy policy);	
	std::string shape() const;
	int size() const;
	size_t bytes() const;
	cudnnTensorDescriptor_t descriptor() const;
	int dim(int i) const;
	std::array<int, 4> dims() const;
	void offload_data();	
	float * cpu_data();
	float * gpu_data();	
	void reset();
	void release();
		
	int is_valid();
	void statistics(double *mean, double *std, double *min, double *max);	
	void print();

	void set(const std::vector<float> &values);	

	bool verify(const std::vector<float> &values);

	float to_float();
	std::shared_ptr<std::vector<float>> to_vec();

	std::string toString();
	std::string name() const;
	
	static size_t used_gpu();

protected:
	std::array<int, 4> _dims;
	size_t _size = 0;
	size_t _bytes = 0;
	cudnnTensorDescriptor_t _desc = 0;	
	std::string _shapeString;
	float *_gpu_data = nullptr;
	float *_cpu_data = nullptr;
	std::string _name;
	DataLocation _location = CPU;
	DataPolicy _policy;
	std::shared_ptr<Tensor> _shadow_tensor;	
	cudaStream_t _stream = nullptr;
	cudaEvent_t _offload_event = nullptr;
	static size_t _used_gpu_mem_size;
};

