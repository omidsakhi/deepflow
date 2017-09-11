#include "nodes/lifting.h"

__global__
void LiftDownKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in_hw = IH * IW;
		int in_chw = IC * in_hw;
		int in_chw_r = i % in_chw;
		int in = (i - in_chw_r) / in_chw;
		int in_hw_r = in_chw_r % in_hw;
		int ic = (in_chw_r - in_hw_r) / in_hw;
		int iw = in_hw_r % IW;
		int ih = (in_hw_r - iw) / IW;		
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
		y[oi] = x[i];
	}	
}

__global__
void LiftUpKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in_hw = IH * IW;
		int in_chw = IC * in_hw;
		int in_chw_r = i % in_chw;
		int in = (i - in_chw_r) / in_chw;
		int in_hw_r = in_chw_r % in_hw;
		int ic = (in_chw_r - in_hw_r) / in_hw;
		int iw = in_hw_r % IW;
		int ih = (in_hw_r - iw) / IW;
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
		y[in * (OC * OW *OH) + oc * (OW * OH) + oh * OW + ow] = x[i];
	}		
}

Lifting::Lifting(deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_lifting_param() == false) << "param.lifting_param() == false";
}

void Lifting::initForward()
{
	_mode = this->_param->lifting_param().mode();

	if (_mode == deepflow::LiftingParam_Mode_UP) {
		auto dims = _inputs[0]->value()->dims();
		LOG_IF(FATAL, dims[1] % 4 != 0) << "Input channels must be a multiple of 4.";		
		_outputs[0]->initValue({ dims[0] , dims[1] / 4, dims[2] * 2, dims[3] * 2 });
		LOG(INFO) << "Lifting Up " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::LiftingParam_Mode_DOWN) {
		auto dims = _inputs[0]->value()->dims();
		LOG_IF(FATAL, dims[2] % 2 != 0) << "Input height must be a multiple of 2.";
		LOG_IF(FATAL, dims[3] % 2 != 0) << "Input width must be a multiple of 2.";
		_outputs[0]->initValue({ dims[0] , dims[1] * 4, dims[2] / 2, dims[3] / 2 });
		LOG(INFO) << "Lifting Down " << _name << " - " << _outputs[0]->value()->shape();
	}	
}

void Lifting::initBackward()
{
	_outputs[0]->initDiff();
}

void Lifting::forward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	if (_mode == deepflow::LiftingParam_Mode_DOWN)
		LiftDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3]);
	else
		LiftUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3]);
	DF_KERNEL_CHECK();
}

void Lifting::backward()
{
	auto size = _outputs[0]->diff()->size();
	auto dims = _outputs[0]->dims();
	if (_mode == deepflow::LiftingParam_Mode_DOWN)
		LiftUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3] );
	else
		LiftDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3] );
	DF_KERNEL_CHECK();
}

std::string Lifting::to_cpp() const
{
	return std::string();
}
