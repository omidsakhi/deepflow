#include "nodes/lifting.h"


__global__
void LiftDownRegularKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, float beta)
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

		int ow = iw / 2;
		int oh = ih / 2;
		bool ceven = (iw % 2 == 0);
		bool reven = (ih % 2 == 0);
		int OW = IW / 2;
		int OH = IH / 2;
		int OC = IC * 4;
		int ch4 = 0;
		if (ceven && reven)
			ch4 = 2;
		else if (ceven && !reven)
			ch4 = 1;
		else if (!ceven && reven)
			ch4 = 3;		
		int oi = in * (OC * OW * OH) + (ic * 4 + ch4) * OW * OH + oh * OW + ow;		
		y[oi] = beta * y[oi] + x[i];
	}	
}

__global__
void LiftUpRegularKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, float beta)
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

		int fc = ic % 4;
		int oc = (ic - fc) / 4;
		int ow, oh;
		if (fc == 0) {
			ow = 2 * iw + 1;
			oh = 2 * ih + 1;
		}
		else if (fc == 1) {			
			ow = 2 * iw;
			oh = 2 * ih + 1;
		}
		else if (fc == 2) {
			ow = 2 * iw;
			oh = 2 * ih;
		}
		else if (fc == 3) {
			ow = 2 * iw + 1;
			oh = 2 * ih;
		}
		int OW = IW * 2;
		int OH = IH * 2;
		int OC = IC / 4;
		int tmp = in * (OC * OW *OH) + oc * (OW * OH) + oh * OW + ow;
		y[tmp] = beta * y[tmp] + x[i];
	}		
}


__global__
void LiftDownFlipKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, float beta)
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

		int ow = iw / 2;
		int oh = ih / 2;
		bool ceven = (iw % 2 == 0);
		bool reven = (ih % 2 == 0);
		int OW = IW / 2;
		int OH = IH / 2;
		int OC = IC * 4;
		int ch4 = 0;
		if (ceven && reven) {
			ch4 = 2;
			oh = OH - oh - 1;
		}
		else if (ceven && !reven) {
			ch4 = 1;
			ow = OW - ow - 1;
		}
		else if (!ceven && reven) {
			ch4 = 3;
			ow = OW - ow - 1;
			oh = OH - oh - 1;
		}
		int oi = in * (OC * OW * OH) + (ic * 4 + ch4) * OW * OH + oh * OW + ow;
		y[oi] = beta * y[oi] +  x[i];
	}
}

__global__
void LiftUpFlipKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, float beta)
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
		int OW = IW * 2;
		int OH = IH * 2;
		int OC = IC / 4;

		int fc = ic % 4;
		int oc = (ic - fc) / 4;
		int ow, oh;
		if (fc == 0) {
			ow = 2 * iw + 1;
			oh = 2 * ih + 1;
		}
		else if (fc == 1) {
			ow = 2 * (IW - iw - 1);
			oh = 2 * ih + 1;
		}
		else if (fc == 2) {
			ow = 2 * iw;
			oh = 2 * (IH - ih - 1);
		}
		else if (fc == 3) {
			ow = 2 * (IW - iw - 1) + 1;
			oh = 2 * (IH - ih - 1);
		}
		int tmp = in * (OC * OW *OH) + oc * (OW * OH) + oh * OW + ow;
		y[tmp] = beta * y[tmp] + x[i];
	}
}

Lifting::Lifting(deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_lifting_param() == false) << "param.lifting_param() == false";
}

void Lifting::init()
{
	_mode = this->_param->lifting_param().mode();	

	if (_mode == deepflow::LiftingParam_Mode_UP_FLIP || _mode == deepflow::LiftingParam_Mode_UP_REGULAR) {
		auto dims = _inputs[0]->value()->dims();
		LOG_IF(FATAL, dims[1] % 4 != 0) << "Input channels must be a multiple of 4.";		
		_outputs[0]->initValue({ dims[0] , dims[1] / 4, dims[2] * 2, dims[3] * 2 });		
	}
	else if (_mode == deepflow::LiftingParam_Mode_DOWN_REGULAR || _mode == deepflow::LiftingParam_Mode_DOWN_FLIP) {
		auto dims = _inputs[0]->value()->dims();
		LOG_IF(FATAL, dims[2] % 2 != 0) << "Input height must be a multiple of 2.";
		LOG_IF(FATAL, dims[3] % 2 != 0) << "Input width must be a multiple of 2.";
		_outputs[0]->initValue({ dims[0] , dims[1] * 4, dims[2] / 2, dims[3] / 2 });		
	}
	else {
		LOG(FATAL);
	}

	_outputs[0]->initDiff();
}

void Lifting::forward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	if (_mode == deepflow::LiftingParam_Mode_DOWN_REGULAR)
		LiftDownRegularKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
	else if (_mode == deepflow::LiftingParam_Mode_DOWN_FLIP)
		LiftDownFlipKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
	else if (_mode == deepflow::LiftingParam_Mode_UP_REGULAR)
		LiftUpRegularKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
	else if (_mode == deepflow::LiftingParam_Mode_UP_FLIP)
		LiftUpFlipKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
	else 
		LOG(FATAL);
	DF_KERNEL_CHECK();
}

void Lifting::backward()
{
	if (_inputs[0]->diff()) {
		auto size = _outputs[0]->diff()->size();
		auto dims = _outputs[0]->dims();
		if (_mode == deepflow::LiftingParam_Mode_DOWN_REGULAR)
			LiftUpRegularKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
		else if (_mode == deepflow::LiftingParam_Mode_DOWN_FLIP)
			LiftUpFlipKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
		else if (_mode == deepflow::LiftingParam_Mode_UP_REGULAR)
			LiftDownRegularKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
		else if (_mode == deepflow::LiftingParam_Mode_UP_FLIP)
			LiftDownFlipKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], 0);
		else
			LOG(FATAL);
		DF_KERNEL_CHECK();
	}
}

std::string Lifting::to_cpp() const
{
	return std::string();
}
