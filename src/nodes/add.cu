#include "core/common_cu.h"

#include "nodes/add.h"

__global__
void AddKernelForward(const int n, const float alpha, const float *a, const float beta, const float *b, float *c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) c[i] = alpha * a[i] + beta * b[i];
}

__global__
void AddKernelBackward(const int n, const float *dy, const float scale, float * __restrict__ dx)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		dx[i] = scale * dy[i];		
	}
}

Add::Add(const deepflow::NodeParam &param) : Node(param) {
	LOG_IF(FATAL, param.has_add_param() == false) << "param.has_add_param() == false";	
}

void Add::initForward() {
	
	auto a = _inputs[0];
	auto ad = a->dims();

	auto b = _inputs[1];
	auto bd = b->dims();

	_alpha = _param.add_param().alpha();
	_beta = _param.add_param().beta();

	LOG_IF(FATAL, a->value()->size() != b->value()->size()) << "Different input sizes";		
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	LOG(INFO) << "Initializing Add " << _name << " - " << _outputs[0]->value()->shape();
}

void Add::initBackward() {
	_outputs[0]->initDiff();
}

void Add::forward() {	
	// C(m,n) = A(m,n) + B(m,n)	
	auto size = _outputs[0]->value()->size();
	AddKernelForward << <numOfBlocks(size), maxThreadsPerBlock >> >(size , _alpha, (float*)_inputs[0]->value()->data(),_beta, (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void Add::backward() {	
	auto size = _outputs[0]->diff()->size();
	if (_inputs[0]->connectedNode()->shouldBackward()) {
		AddKernelBackward << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), _alpha, (float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}	
	if (_inputs[1]->connectedNode()->shouldBackward()) {
		AddKernelBackward << <numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), _beta, (float*)_inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string Add::to_cpp() const
{
	std::string op;
	float print_alpha_beta = false;
	if (_alpha == 1 && _beta == 1)
		op = "add";
	else if (_alpha == 1 && _beta == -1)
		op = "subtract";
	else {
		op = "add";
		print_alpha_beta = true;
	}
	std::string cpp = "auto " + _name + " = df." + op + "(" + _inputs[0]->connectedNode()->name() + ", " + _inputs[1]->connectedNode()->name() + ", ";
	if (print_alpha_beta)
		cpp += std::to_string(_alpha) + ", " + std::to_string(_beta);
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
