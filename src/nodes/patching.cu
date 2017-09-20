#include "nodes/patching.h"

__global__
void PatchingDownKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int PH, int PW)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in_hw = IH * IW;
		int in_chw = IC * in_hw;
		int in_chw_r = i % in_chw;
		int in = (i - in_chw_r) / in_chw; // sample #
		int in_hw_r = in_chw_r % in_hw;
		int ic = (in_chw_r - in_hw_r) / in_hw; // channel #
		int icolumn = in_hw_r % IW; // input column
		int irow = (in_hw_r - icolumn) / IW; // input row
		int pcp = icolumn % PW; // pixel column in patch
		int prp = irow % PH; // pixel row in patch
		int pcolumn = (icolumn - pcp) / PW; // patch column
		int prow = (irow-prp) / PH; // patch row
		int THP = IH / PH; // total number of patches h
		int TWP = IW / PW; // total number of patches w
		int OC = IC * THP * TWP;
		int oi = in * (OC * PW * PH) + (ic * THP * TWP + prow * TWP + pcolumn) * PH * PW + prp * PW + pcp;
		y[oi] = x[i];
	}
}

__global__
void PatchingUpKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int THP, int TWP)
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
		int OC = IC / THP / TWP;
		int OH = IH * THP;
		int OW = IW * TWP;
		int fc = ic % (THP * TWP);
		int oc = (ic - fc) / (THP*TWP);
		int ph = fc % TWP;
		int pw = (fc - ph) / TWP;
		int oh = pw * IH + ih;
		int ow = ph * IW + iw;
		y[in * (OC * OW *OH) + oc * (OW * OH) + oh * OW + ow] = x[i];
	}
}

Patching::Patching(deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_patching_param() == false) << "param.patching_param() == false";
}

void Patching::initForward() 
{
	// NCHW
	_mode = this->_param->patching_param().mode();
	_num_horizontal_patches = this->_param->patching_param().num_horizontal_patch();
	_num_vertical_patches = this->_param->patching_param().num_vertical_patch();
	auto dims = _inputs[0]->value()->dims();	
	int input_width = dims[3];
	int input_height = dims[2];
	if (_mode == deepflow::PatchingParam_Mode_UP) {		
		LOG_IF(FATAL, dims[1] % (_num_horizontal_patches * _num_vertical_patches) != 0) << "Input channels must be a multiple of number of patches" << (_num_horizontal_patches * _num_vertical_patches);
		_outputs[0]->initValue({ dims[0] , dims[1] / (_num_horizontal_patches * _num_vertical_patches), input_height * _num_vertical_patches, input_width * _num_horizontal_patches});
		LOG(INFO) << "Patching Up " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::PatchingParam_Mode_DOWN) {
		LOG_IF(FATAL, dims[2] % _num_vertical_patches != 0) << _name << " Input height must be a multiple of number of vertical patchs " << _num_vertical_patches;
		LOG_IF(FATAL, dims[3] % _num_horizontal_patches != 0) << _name << " Input width must be a multiple of number of horizontal patchs " << _num_horizontal_patches;
		int patch_h = input_height / _num_vertical_patches;
		int patch_w = input_width / _num_horizontal_patches;
		_outputs[0]->initValue({ dims[0] , dims[1] * _num_horizontal_patches * _num_vertical_patches, patch_h, patch_w });
		LOG(INFO) << "Patching Down " << _name << " - " << _outputs[0]->value()->shape();
	}
}

void Patching::initBackward()
{
	_outputs[0]->initDiff();
}

void Patching::forward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	if (_mode == deepflow::PatchingParam_Mode_DOWN) {
		int patch_h = dims[2] / _num_vertical_patches;
		int patch_w = dims[3] / _num_horizontal_patches;
		PatchingDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], patch_h, patch_w);
	}
	else
		PatchingUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], _num_vertical_patches, _num_horizontal_patches);
	DF_KERNEL_CHECK();
}

void Patching::backward()
{
	auto size = _outputs[0]->diff()->size();
	auto dims = _outputs[0]->dims();
	if (_mode == deepflow::PatchingParam_Mode_DOWN)
		PatchingUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], _num_vertical_patches, _num_horizontal_patches);
	else {
		int patch_h = dims[2] / _num_vertical_patches;
		int patch_w = dims[3] / _num_horizontal_patches;
		PatchingDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], patch_h, patch_w);
	}
	DF_KERNEL_CHECK();
}

std::string Patching::to_cpp() const
{
	return std::string();
}
