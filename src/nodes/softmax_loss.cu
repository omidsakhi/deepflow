#include "core/common_cu.h"

#include "nodes/softmax_loss.h"

__global__
void SoftmaxLossKernelBackward(const int n, const float *softmax_output,const float * __restrict__ target, float * __restrict__ diff)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) diff[i] = target[i] - softmax_output[i];
}


SoftmaxLoss::SoftmaxLoss(const deepflow::NodeParam &param) : Loss(param) {
	LOG_IF(FATAL, param.loss_param().has_softmax_loss_param() == false) << "param.loss_param().has_softmax_loss_param() == false";
}

void SoftmaxLoss::initForward() {		
	LOG(INFO) << "Initializing SoftmaxLoss " << _name << " - " << _inputs[0]->value()->shape();
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input size != target size";
	DF_CUDNN_CHECK(cudnnCreate(&_cudnnHandle));	
	_outputs[0]->initValue(_inputs[0]->value()->dims());
	_outputs[1]->initValue(_inputs[0]->value()->dims());
}

void SoftmaxLoss::initBackward() {
	_outputs[0]->initDiff();
	_outputs[1]->initDiff();
}

void SoftmaxLoss::forward() {
	DF_CUDNN_CHECK(cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &beta, _outputs[0]->value()->descriptor(), _outputs[1]->value()->mutableData()));
}

void SoftmaxLoss::backward() {
	size_t size = _outputs[0]->value()->size();
	SoftmaxLossKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*) _outputs[1]->value()->data(), (float*) _inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
	DF_CUDA_CHECK(cudaMemcpy(_inputs[0]->diff()->mutableData(), _outputs[0]->value()->data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToDevice));	
}

std::string SoftmaxLoss::to_cpp() const
{	
	std::string cpp = "auto " + _name + " = df.softmax_loss(" + _inputs[0]->connectedNode()->name() + ", " + _inputs[1]->connectedNode()->name() + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
