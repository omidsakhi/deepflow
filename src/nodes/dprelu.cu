#include "nodes/dprelu.h"

#include "core/common_cu.h"

__global__
void DPReluForwardKernel(int size, int input_channels, int input_width, int input_height, int a_channels, const float *x, const float *a,  float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
	{
		int indexTemp = i;
		const int w = indexTemp % input_width;
		indexTemp /= input_width;
		const int h = indexTemp % input_height;
		indexTemp /= input_height;
		const int c = indexTemp % input_channels;
		indexTemp /= input_channels;
		const int n = indexTemp;
		const int ia = n * a_channels + c;
		float _a = a[ia];
		_a = _a > 0 ? 1 : 0;		
		if (x[i] > 0)
			y[i] = x[i] * _a;
		else {			
			y[i] = 0.2f * x[i] * _a;
		}
	}
}

__global__
void DPReluBackwardKernel(int size, int input_channels, int input_width, int input_height, int a_channels, const float *x, const float *a, const float * dy, float *dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < size)
	{
		int indexTemp = i;
		const int w = indexTemp % input_width;
		indexTemp /= input_width;
		const int h = indexTemp % input_height;
		indexTemp /= input_height;
		const int c = indexTemp % input_channels;
		indexTemp /= input_channels;
		const int n = indexTemp;
		const int ia = n * a_channels + c;
		float _a = a[ia];
		_a = _a > 0 ? 1 : 0;
		if (x[i] > 0)
			dx[i] = dy[i] * _a;
		else {			
			dx[i] = 0.2f * dy[i] * _a;
		}
	}
}


DPRelu::DPRelu(deepflow::NodeParam * param) : Node(param)
{
	LOG_IF(FATAL, param->has_dprelu_param() == false) << "param.has_dprelu_param() == false";
}

void DPRelu::init()
{
	auto input_dims = _inputs[0]->dims();
	auto a_dims = _inputs[1]->dims();
	LOG_IF(FATAL, a_dims[0] != input_dims[0]) << "[FAILED] " << _name << " : batch sizes must be the same";
	LOG_IF(FATAL, a_dims[1] < input_dims[1]) << "[FAILED] " << _name;	
	_outputs[0]->initValue(input_dims);
	_outputs[0]->initDiff();
}

void DPRelu::forward()
{
	auto input_dims = _inputs[0]->dims();
	auto a_channels = _inputs[1]->value()->dim(1);
	auto size = _inputs[0]->value()->size();
	DPReluForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> >
		(size, input_dims[1], input_dims[2], input_dims[3], a_channels, _inputs[0]->value()->gpu_data(DF_LINE), _inputs[1]->value()->gpu_data(DF_LINE), (float*)_outputs[0]->value()->gpu_data(DF_LINE));
	DF_NODE_KERNEL_CHECK();
}

void DPRelu::backward()
{
	if (_inputs[0]->diff()) {
		auto input_dims = _inputs[0]->dims();
		auto a_channels = _inputs[1]->value()->dim(1);
		auto size = _inputs[0]->value()->size();
		DPReluBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> >
			(size, input_dims[1], input_dims[2], input_dims[3], a_channels, _inputs[0]->value()->gpu_data(DF_LINE), _inputs[1]->value()->gpu_data(DF_LINE), _outputs[0]->diff()->gpu_data(DF_LINE), (float*)_inputs[0]->diff()->gpu_data(DF_LINE));
		DF_NODE_KERNEL_CHECK();
	}
}

std::string DPRelu::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.dprelu(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;	
}
