#include "core/common_cu.h"

#include "ops/softmax_loss.h"

#include <glog/logging.h>

__global__
void SoftmaxLossKernelBackward(const int n, const float *softmax_output,const float * __restrict__ target, float * __restrict__ diff)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) diff[i] = target[i] - softmax_output[i];
}


SoftmaxLoss::SoftmaxLoss(NodeParam param ) : Node(param) {
	LOG_IF(FATAL, param.loss_param().has_softmax_loss_param() == false);
}

void SoftmaxLoss::initForward() {		
	LOG(INFO) << "Initializing SoftmaxLoss (name: " << _name << " ) | Shape : " << _inputs[0]->value()->toString();
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input size != target size";
	LOG_IF(FATAL, cudnnCreate(&_cudnnHandle) != 0);	
	_outputs[0]->initValue(_inputs[0]->value()->dims());	
}

void SoftmaxLoss::initBackward() {
	_outputs[0]->initDiff();
}

void SoftmaxLoss::forward() {
	LOG_IF(FATAL, cudnnSoftmaxForward(_cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, _inputs[0]->value()->descriptor(), _inputs[0]->value()->data(), &beta, _outputs[0]->value()->descriptor(), _outputs[0]->value()->mutableData()) != 0);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);	
}

void SoftmaxLoss::backward() {
	size_t size = _outputs[0]->value()->size();
	SoftmaxLossKernelBackward << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*) _outputs[0]->value()->data(), (float*) _inputs[1]->value()->data(), (float*)_inputs[0]->diff()->mutableData());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);
	auto value = _inputs[0]->diff()->cpyToHost<float>();
	double sum = 0.0;
	for (int i = 0; i < value->size(); ++i)
		sum += value->at(i) * value->at(i);
	sum /= value->size();
	LOG(INFO) << "LOSS: " << sum;
	if (sum != sum) {
		//_inputs[0]->value()->print<float>();
	}
	LOG_IF(FATAL, sum != sum);	
}