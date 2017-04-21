
#include "core/common_cu.h"

#include "ops/euclidean_loss.h"

__global__
void EuclideanLossKernel(const int n, const float * __restrict__ x1, const float * __restrict__ x2, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		y[i] = x1[i] - x2[i];
}

EuclideanLoss::EuclideanLoss(const NodeParam &param) : Loss(param) {
	LOG_IF(FATAL, param.loss_param().has_euclidean_loss_param() == false) << "param.loss_param().has_euclidean_loss_param() == false";
}

void EuclideanLoss::initForward() {
	LOG(INFO) << "Initializing EuclideanLoss " << _name << " - " << _inputs[0]->value()->shape();
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input " << _inputs[0]->value()->shape() << " != " << " Target " << _inputs[1]->value()->shape();
	_outputs[0]->initValue(_inputs[0]->value()->dims());	
}

void EuclideanLoss::initBackward() {	
	_outputs[0]->initDiff();	
}

void EuclideanLoss::forward() {
	auto size = _inputs[0]->value()->size();
	EuclideanLossKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void EuclideanLoss::backward() {	
	DF_CUDA_CHECK(cudaMemcpy(_inputs[0]->diff()->mutableData(), _outputs[0]->value()->data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice));
}
