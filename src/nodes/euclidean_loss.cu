
#include "core/common_cu.h"

#include "nodes/euclidean_loss.h"

__global__
void EuclideanLossKernel(const int n, const float * __restrict__ x1, const float * __restrict__ x2, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		y[i] += x2[i] - x1[i];
}

EuclideanLoss::EuclideanLoss(const deepflow::NodeParam &param) : Loss(param) {
	LOG_IF(FATAL, param.loss_param().has_euclidean_loss_param() == false) << "param.loss_param().has_euclidean_loss_param() == false";
}

void EuclideanLoss::initForward() {
	LOG(INFO) << "Initializing EuclideanLoss " << _name << " - " << _inputs[0]->value()->shape();
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input " << _inputs[0]->value()->shape() << " != " << " Target " << _inputs[1]->value()->shape();	
}

void EuclideanLoss::initBackward() {
}

void EuclideanLoss::forward() {
	
}

void EuclideanLoss::backward() {		
	auto size = _inputs[0]->value()->size();
	EuclideanLossKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_inputs[0]->diff()->mutableData());
	DF_KERNEL_CHECK();
}

std::string EuclideanLoss::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.euclidean_loss(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
