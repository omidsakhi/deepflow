#include "nodes/Upsample.h"

__global__
void UpsampleBackwardKernel(const int n, const float *dY, float *dX, int IN, int IC, int IH, int IW, int OH, int OW, int OWN, int OWC)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in = i;
		int iw = i % IW;
		in /= IW;
		int ih = in % IH;
		in /= IH;
		int ic = in % IC;
		in /= IC;

		int ow, oh, oi;
		int offset = in * OWN + ic * OWC;
		ow = 2 * iw + 1;
		oh = 2 * ih + 1;		
		oi = offset + oh * OW + ow;
		float tmp = dY[oi];		
		ow = 2 * iw;
		oh = 2 * ih + 1;
		oi = offset + oh * OW + ow;
		tmp += dY[oi];		
		ow = 2 * iw;
		oh = 2 * ih;
		oi = offset + oh * OW + ow;
		tmp += dY[oi];		
		ow = 2 * iw + 1;
		oh = 2 * ih;
		oi = offset + oh * OW + ow;
		tmp += dY[oi];
		dX[i] = tmp;
	}	
}

__global__
void UpsampleForwardKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int OH, int OW, int OWN, int OWC)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in = i;
		int iw = i % IW;
		in /= IW;
		int ih = in % IH;
		in /= IH;
		int ic = in % IC;
		in /= IC;

		int ow, oh, oi;
		int offset = in * OWN + ic * OWC;
		ow = 2 * iw + 1;
		oh = 2 * ih + 1;
		oi = offset + oh * OW + ow;
		y[oi] = x[i];
		ow = 2 * iw;
		oh = 2 * ih + 1;
		oi = offset + oh * OW + ow;
		y[oi] = x[i];
		ow = 2 * iw;
		oh = 2 * ih;
		oi = offset + oh * OW + ow;
		y[oi] = x[i];
		ow = 2 * iw + 1;
		oh = 2 * ih;
		oi = offset + oh * OW + ow;
		y[oi] = x[i];
	}		
}

Upsample::Upsample(deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_upsample_param() == false) << "param.upsample_param() == false";
}

void Upsample::initForward()
{
	auto dims = _inputs[0]->value()->dims();	
	_outputs[0]->initValue({ dims[0] , dims[1], dims[2] * 2, dims[3] * 2 });
	LOG(INFO) << "Upsample Up " << _name << " - " << _outputs[0]->value()->shape();
}

void Upsample::initBackward()
{
	_outputs[0]->initDiff();
}

void Upsample::forward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	UpsampleForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > 
		(size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], dims[2] * 2, dims[3] * 2,
			dims[1] * dims[2] * dims[3] * 4,
			dims[2] * dims[3] * 4);
	DF_KERNEL_CHECK();
}

void Upsample::backward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	UpsampleBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > 
		(size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], dims[2] * 2, dims[3] * 2,
		dims[1] * dims[2] * dims[3] * 4,
			dims[2] * dims[3] * 4
		);
	DF_KERNEL_CHECK();
}

std::string Upsample::to_cpp() const
{
	return std::string();
}
