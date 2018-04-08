#include "nodes/patching.h"

__global__
void PatchingDownKernel(const int n, bool down_samples, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int PH, int PW, float beta)
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
		if (down_samples) {
			int pcp = iw % PW; // pixel column in patch
			int prp = ih % PH; // pixel row in patch
			int pcolumn = (iw - pcp) / PW; // patch column
			int prow = (ih - prp) / PH; // patch row
			int THP = IH / PH; // total number of patches h
			int TWP = IW / PW; // total number of patches w
			int oi = (in * THP * TWP + prow * TWP + pcolumn) * (IC * PW * PH) + ic * PH * PW + prp * PW + pcp;
			y[oi] = beta * y[oi] + x[i];
		}
		else {
			int pcp = iw % PW; // pixel column in patch
			int prp = ih % PH; // pixel row in patch
			int pcolumn = (iw - pcp) / PW; // patch column
			int prow = (ih - prp) / PH; // patch row
			int THP = IH / PH; // total number of patches h
			int TWP = IW / PW; // total number of patches w
			int OC = IC * THP * TWP;
			int oi = in * (OC * PW * PH) + (ic * THP * TWP + prow * TWP + pcolumn) * PH * PW + prp * PW + pcp;
			y[oi] = beta * y[oi] + x[i];
		}		
	}
}

__global__
void PatchingUpKernel(const int n, bool up_samples, const float * __restrict__ x, float * __restrict__ y, int IN, int IC, int IH, int IW, int THP, int TWP, float beta)
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
		
		if (up_samples) {
			int OH = IH * THP;
			int OW = IW * TWP;
			int fn = in % (THP * TWP);
			int on = (in - fn) / (THP*TWP);

			int ph = fn % TWP;
			int pw = (fn - ph) / TWP;
			int oh = pw * IH + ih;
			int ow = ph * IW + iw;
			int tmp = on * (IC * OW *OH) + ic * (OW * OH) + oh * OW + ow;
			y[tmp] = beta * y[tmp] + x[i];
		}
		else {
			int OC = IC / THP / TWP;
			int OH = IH * THP;
			int OW = IW * TWP;
			int fc = ic % (THP * TWP);
			int oc = (ic - fc) / (THP*TWP);
			int ph = fc % TWP;
			int pw = (fc - ph) / TWP;
			int oh = pw * IH + ih;
			int ow = ph * IW + iw;
			int tmp = in * (OC * OW *OH) + oc * (OW * OH) + oh * OW + ow;
			y[tmp] = beta * y[tmp] + x[i];
		}			 

	}
}

Patching::Patching(deepflow::NodeParam * param) : Node(param) {
	LOG_IF(FATAL, param->has_patching_param() == false) << "param.patching_param() == false";
}

void Patching::init() 
{
	// NCHW
	_mode = this->_param->patching_param().mode();
	_num_horizontal_patches = this->_param->patching_param().num_horizontal_patch();
	_num_vertical_patches = this->_param->patching_param().num_vertical_patch();
	auto dims = _inputs[0]->value()->dims();	
	int input_width = dims[3];
	int input_height = dims[2];
	if (_mode == deepflow::PatchingParam_Mode_UPSAMPLES) {		
		LOG_IF(FATAL, dims[0] % (_num_horizontal_patches * _num_vertical_patches) != 0) << "Input samples must be a multiple of number of patches" << (_num_horizontal_patches * _num_vertical_patches);
		_outputs[0]->initValue({ dims[0] / (_num_horizontal_patches * _num_vertical_patches), dims[1], input_height * _num_vertical_patches, input_width * _num_horizontal_patches});
		//LOG(INFO) << "Patching Up " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::PatchingParam_Mode_UPCHANNELS) {
		LOG_IF(FATAL, dims[1] % (_num_horizontal_patches * _num_vertical_patches) != 0) << "Input channels must be a multiple of number of patches" << (_num_horizontal_patches * _num_vertical_patches);
		_outputs[0]->initValue({ dims[0], dims[1] / (_num_horizontal_patches * _num_vertical_patches), input_height * _num_vertical_patches, input_width * _num_horizontal_patches });
		//LOG(INFO) << "Patching Up " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES) {
		LOG_IF(FATAL, dims[2] % _num_vertical_patches != 0) << _name << " Input height must be a multiple of number of vertical patchs " << _num_vertical_patches;
		LOG_IF(FATAL, dims[3] % _num_horizontal_patches != 0) << _name << " Input width must be a multiple of number of horizontal patchs " << _num_horizontal_patches;
		int patch_h = input_height / _num_vertical_patches;
		int patch_w = input_width / _num_horizontal_patches;
		_outputs[0]->initValue({ dims[0] * _num_horizontal_patches * _num_vertical_patches, dims[1], patch_h, patch_w });
		//LOG(INFO) << "Patching Down " << _name << " - " << _outputs[0]->value()->shape();
	}
	else if (_mode == deepflow::PatchingParam_Mode_DOWNCHANNELS) {
		LOG_IF(FATAL, dims[2] % _num_vertical_patches != 0) << _name << " Input height must be a multiple of number of vertical patchs " << _num_vertical_patches;
		LOG_IF(FATAL, dims[3] % _num_horizontal_patches != 0) << _name << " Input width must be a multiple of number of horizontal patchs " << _num_horizontal_patches;
		int patch_h = input_height / _num_vertical_patches;
		int patch_w = input_width / _num_horizontal_patches;
		_outputs[0]->initValue({ dims[0], dims[1] * _num_horizontal_patches * _num_vertical_patches, patch_h, patch_w });
		//LOG(INFO) << "Patching Down " << _name << " - " << _outputs[0]->value()->shape();
	}
	else
		LOG(FATAL);

	_outputs[0]->initDiff();
}

void Patching::forward()
{
	auto size = _inputs[0]->value()->size();
	auto dims = _inputs[0]->dims();
	if (_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES || _mode == deepflow::PatchingParam_Mode_DOWNCHANNELS) {
		int patch_h = dims[2] / _num_vertical_patches;
		int patch_w = dims[3] / _num_horizontal_patches;
		PatchingDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size,  
			_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES,
			(float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], patch_h, patch_w, 0);
	}
	else
		PatchingUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size,
			_mode == deepflow::PatchingParam_Mode_UPSAMPLES,
		(float*)_inputs[0]->value()->data(), (float*)_outputs[0]->value()->mutableData(), dims[0], dims[1], dims[2], dims[3], _num_vertical_patches, _num_horizontal_patches, 0);
	DF_KERNEL_CHECK();
}

void Patching::backward()
{
	if (_inputs[0]->diff()) {
		auto size = _outputs[0]->diff()->size();
		auto dims = _outputs[0]->dims();
		if (_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES || _mode == deepflow::PatchingParam_Mode_DOWNCHANNELS)
			PatchingUpKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size,
				_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES,
				(float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], _num_vertical_patches, _num_horizontal_patches, 0);
		else {
			int patch_h = dims[2] / _num_vertical_patches;
			int patch_w = dims[3] / _num_horizontal_patches;
			PatchingDownKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size,
				_mode == deepflow::PatchingParam_Mode_UPSAMPLES,
				(float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData(), dims[0], dims[1], dims[2], dims[3], patch_h, patch_w, 0);
		}
		DF_KERNEL_CHECK();
	}
}

std::string Patching::to_cpp() const
{
	// std::string patching(std::string input, PatchingMode mode, int num_vertical_patches, int num_horizontal_patches, std::string name = "patching", std::initializer_list<std::string> phases = {});
	std::string cpp = "auto " + _name + " = df.patching(" + _input_name_for_cpp(0) + ", ";
	if (_mode == deepflow::PatchingParam_Mode_DOWNSAMPLES) 
	{
		cpp += "DeepFlow::PATCHING_DOWNSAMPLES, ";
	} 
	else if (_mode == deepflow::PatchingParam_Mode_DOWNCHANNELS) {
		cpp += "DeepFlow::PATCHING_DOWNCHANNELS, ";
	}
	else if (_mode == deepflow::PatchingParam_Mode_UPSAMPLES) {
		cpp += "DeepFlow::PATCHING_UPSAMPLES, ";
	}
	else if (_mode == deepflow::PatchingParam_Mode_UPCHANNELS) {
		cpp += "DeepFlow::PATCHING_UPCHANNELS, ";
	}
	cpp += "\"" + _name + "\");";	
	return cpp;
}
