#include "nodes/patching.h"

__global__
void PatchingDownKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int PH, int PW)
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
		int pcp = iw % PW; // pixel column in patch
		int prp = ih % PH; // pixel row in patch
		int pcolumn = (iw - pcp) / PW; // patch column
		int prow = (ih-prp) / PH; // patch row
		int THP = IH / PH; // total number of patches h
		int TWP = IW / PW; // total number of patches w
		int oi = (in * THP * TWP + prow * TWP + pcolumn) * (IC * PW * PH) + ic * PH * PW + prp * PW + pcp;
		y[oi] = x[i];
	}
}

__global__
void PatchingUpKernel(const int n, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int THP, int TWP)
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
		
		int OH = IH * THP;
		int OW = IW * TWP;
		int fn = in % (THP * TWP);
		int on = (in - fn) / (THP*TWP);

		int ph = fn % TWP;
		int pw = (fn - ph) / TWP;
		int oh = pw * IH + ih;
		int ow = ph * IW + iw;
		y[on * (IC * OW *OH) + ic * (OW * OH) + oh * OW + ow] = x[i];
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
		LOG_IF(FATAL, dims[0] % (_num_horizontal_patches * _num_vertical_patches) != 0) << "Input samples must be a multiple of number of patches" << (_num_horizontal_patches * _num_vertical_patches);
		_outputs[0]->initValue({ dims[0] / (_num_horizontal_patches * _num_vertical_patches), dims[1], input_height * _num_vertical_patches, input_width * _num_horizontal_patches});
		LOG(INFO) << "Patching Up " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::PatchingParam_Mode_DOWN) {
		LOG_IF(FATAL, dims[2] % _num_vertical_patches != 0) << _name << " Input height must be a multiple of number of vertical patchs " << _num_vertical_patches;
		LOG_IF(FATAL, dims[3] % _num_horizontal_patches != 0) << _name << " Input width must be a multiple of number of horizontal patchs " << _num_horizontal_patches;
		int patch_h = input_height / _num_vertical_patches;
		int patch_w = input_width / _num_horizontal_patches;
		_outputs[0]->initValue({ dims[0] * _num_horizontal_patches * _num_vertical_patches, dims[1], patch_h, patch_w });
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
