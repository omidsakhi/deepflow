#include "nodes/log.h"
#include "core/common_cu.h"

__global__
void LogKernelForward(const int n, const float coef, const float * __restrict__ x, float * __restrict__ x2)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		x2[i] = coef * log(x[i]);
}

__global__
void LogKernelBackward(const int n, const float coef, const float *x, const float *diff, float *out)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = coef * diff[i] / x[i];
}

Log::Log(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_log_param() == false) << "param.has_log_param() == false";
}

void Log::init() {
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[0]->initDiff();
	LOG(INFO) << "Log " << _name << " - " << _outputs[0]->value()->shape();
}

void Log::forward() {
	auto size = _inputs[0]->value()->size();
	auto coef = _param->log_param().coef();
	LogKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, coef, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Log::backward() {
	auto size = _inputs[0]->value()->size();
	auto coef = _param->log_param().coef();
	LogKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, coef, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_KERNEL_CHECK();
}

std::string Log::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.log(" + _input_name_for_cpp(0) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
