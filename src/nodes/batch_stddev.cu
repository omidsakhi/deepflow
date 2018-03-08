#include "nodes\batch_stddev.h"

__global__
void StdDevKernel(const int n, const int inner_dim, const float *in, const float *avg, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {		
		out[i] = (in[i] - avg[i % inner_dim]);
	}
}

BatchStdDev::BatchStdDev(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_batch_stddev_param() == false) << "param.has_batch_stddev_param() == false";
}

void BatchStdDev::init()
{	
	auto inputDims = _inputs[0]->value()->dims();
	DF_NODE_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	DF_NODE_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_avgDesc1));
	DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_avgDesc1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, inputDims[1], inputDims[2], inputDims[3]));
	_inner_dim = inputDims[1] * inputDims[2] * inputDims[3];
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_avg, _inner_dim * sizeof(float)));
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_std, _inner_dim * sizeof(float)));
	DF_NODE_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_avgReduceTensorDesc1));
	DF_NODE_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_avgReduceTensorDesc1, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
	size_t workspace1;
	DF_NODE_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(_cudnnHandle, _avgReduceTensorDesc1, _inputs[0]->value()->descriptor(), _avgDesc1, &workspace1));

	DF_NODE_CUDNN_CHECK(cudnnCreateTensorDescriptor(&_avgDesc2));
	DF_NODE_CUDNN_CHECK(cudnnSetTensor4dDescriptor(_avgDesc2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
	DF_NODE_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&_avgReduceTensorDesc2));
	DF_NODE_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(_avgReduceTensorDesc2, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
	size_t workspace2;
	DF_NODE_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(_cudnnHandle, _avgReduceTensorDesc2, _avgDesc1, _avgDesc2, &workspace2));
	_workspaceSizeInBytes = std::max(workspace1, workspace2);
	DF_NODE_CUDA_CHECK(cudaMalloc(&_d_workspace, _workspaceSizeInBytes));
	
	_outputs[0]->initValue({ 1,1,1,1 });
}

void BatchStdDev::forward()
{
	DF_NODE_CUDNN_CHECK(
		cudnnReduceTensor(
			_cudnnHandle,
			_avgReduceTensorDesc1,
			nullptr,
			0,
			_d_workspace,
			_workspaceSizeInBytes,
			&one,
			_inputs[0]->value()->descriptor(),
			_inputs[0]->value()->data(),
			&zero,
			_avgDesc1,
			_d_avg)
	);	
	float alpha = 1.0f / sqrt(_inner_dim - 1);
	StdDevKernel <<< numOfBlocks(_inner_dim), maxThreadsPerBlock >>> (_inner_dim, _inner_dim, (float*)_inputs[0]->value()->data(), _d_avg,  _d_std);
	DF_NODE_KERNEL_CHECK();
	DF_NODE_CUDNN_CHECK(
		cudnnReduceTensor(
			_cudnnHandle,
			_avgReduceTensorDesc2,
			nullptr,
			0,
			_d_workspace,
			_workspaceSizeInBytes,
			&alpha,
			_avgDesc1,
			_d_std,
			&zero,
			_outputs[0]->value()->descriptor(),
			_outputs[0]->value()->mutableData())
	);
}

void BatchStdDev::backward()
{
	cudaMemset(_inputs[0]->diff()->mutableData(), 0, _inputs[0]->diff()->sizeInBytes());
}

std::string BatchStdDev::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.batch_stddev(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;	
}
