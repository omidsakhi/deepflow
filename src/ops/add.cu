#include "core/common_cu.h"

#include "ops/add.h"

__global__
void AddKernelForward(const int n, const float alpha, const float *a, const float beta, const float *b, float *c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) c[i] = alpha * a[i] + beta * b[i];
}

__global__
void AddKernelBackward(const int n, const float *diff, const float alpha, float * __restrict__ a_diff, const float beta, float * __restrict__ b_diff)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		a_diff[i] = alpha * diff[i];
		b_diff[i] = beta * diff[i];
	}
}

Add::Add(const NodeParam &param) : Node(param) {	
	LOG_IF(FATAL, param.has_add_param() == false);	
}

void Add::initForward() {
	
	auto a = _inputs[0];
	auto ad = a->dims();

	auto b = _inputs[1];
	auto bd = b->dims();

	_alpha = _param.add_param().alpha();
	_beta = _param.add_param().beta();

	LOG_IF(FATAL, a->value()->size() != b->value()->size()) << "Different input sizes";	
	LOG(INFO) << "Initializing Add (name: " << _name << " ) for " << a->node()->name() << " and " << b->node()->name();	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
}

void Add::initBackward() {
	_outputs[0]->initDiff();
}

void Add::forward() {	
	// C(m,n) = A(m,n) + B(m,n)	
	auto size = _outputs[0]->value()->size();
	AddKernelForward << <numOfBlocks(size), maxThreadsPerBlock >> >(size , _alpha, (float*)_inputs[0]->value()->data(),_beta, (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);
}

void Add::backward() {	
	auto size = _outputs[0]->diff()->size();	
	AddKernelBackward << <numOfBlocks(size), maxThreadsPerBlock >> >(size, (float*)_outputs[0]->diff()->data(), _alpha, (float*)_inputs[0]->diff()->mutableData(), _beta, (float*)_inputs[1]->diff()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);

}
