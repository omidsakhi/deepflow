#include "nodes/reduce_all.h"
#include "core/common_cu.h"

__global__
void ReduceAllKernelForward(const int n, bool average, const float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {		
		atomicAdd(&y[0], average ? x[i] / n : x[i]);
	}
}

__global__
void ReduceAllKernelBackward(const int n, bool average, const float *dY, float *dX)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		dX[i] = (average ? dY[0] / n : dY[0]);
	}		
}

ReduceAll::ReduceAll(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_reduce_all_param() == false) << "param.has_reduce_all_param() == false";
	_reduce_op = _param->reduce_all_param().reduce_op();
}

std::string ReduceAll::op_name() const
{	
	std::string op_name = (_reduce_op == deepflow::ReduceAllParam_ReduceAllOp_AVG ? "reduce_mean" : "reduce_sum");
	return op_name;
}

void ReduceAll::init() {
	_outputs[0]->initValue({ 1, 1, 1, 1});	
	_outputs[0]->initDiff();	
}

void ReduceAll::forward() {
	auto size = _inputs[0]->value()->size();
	DF_CUDA_CHECK(cudaMemset(_outputs[0]->value()->gpu_data(DF_LINE), 0, _outputs[0]->value()->bytes()));	
	ReduceAllKernelForward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _reduce_op == deepflow::ReduceAllParam_ReduceAllOp_AVG, _inputs[0]->value()->gpu_data(DF_LINE), (float*)_outputs[0]->value()->gpu_data(DF_LINE));
	DF_KERNEL_CHECK();
}

void ReduceAll::backward() {
	if (_inputs[0]->diff()) {
		auto size = _inputs[0]->value()->size();
		ReduceAllKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, _reduce_op == deepflow::ReduceAllParam_ReduceAllOp_AVG, _outputs[0]->diff()->gpu_data(DF_LINE), (float*)_inputs[0]->diff()->gpu_data(DF_LINE));
		DF_KERNEL_CHECK();
	}
}

std::string ReduceAll::to_cpp() const
{
	return "";
}