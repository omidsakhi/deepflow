#include "nodes/restructure.h"

#include "core/common_cu.h"

__global__
void RestructureKernel(int n, const float *x, const int N, const int C, const int H, const int W, const int swap_dim_1, const int swap_dim_2, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		int in_hw = H * W;
		int in_chw = C * in_hw;
		int in_chw_r = i % in_chw;
		int in_n = (i - in_chw_r) / in_chw;		
		int in_hw_r = in_chw_r % in_hw;
		int in_c = (in_chw_r - in_hw_r) / in_hw;
		int in_w = in_hw_r % W;
		int in_h = (in_hw_r - in_w) / W;
		int o = in_n * C * H * W + in_c * H * W + in_h * W + in_w;
		if ((swap_dim_1 == 0 && swap_dim_2 == 1) || (swap_dim_1 == 1 && swap_dim_2 == 0)) { // n -> c , c -> n
			o = in_c * N * H * W + in_n * H * W + in_h * W + in_w;
		} 
		else if ((swap_dim_1 == 0 && swap_dim_2 == 2) || (swap_dim_1 == 2 && swap_dim_2 == 0)) { // n -> h, h -> n
			o = in_h * C * N * W + in_c * N * W + in_n * W + in_w;
		}
		else if ((swap_dim_1 == 0 && swap_dim_2 == 3) || (swap_dim_1 == 3 && swap_dim_2 == 0)) { // n -> w, w -> n
			o = in_w * C * H * N + in_c * H * N + in_h * N + in_n;
		}
		else if ((swap_dim_1 == 1 && swap_dim_2 == 2) || (swap_dim_1 == 2 && swap_dim_2 == 1)) { // c -> h, h -> c
			o = in_n * H * C * W + in_h * C * W + in_c * W + in_w;
		}
		else if ((swap_dim_1 == 1 && swap_dim_2 == 3) || (swap_dim_1 == 3 && swap_dim_2 == 1)) { // c -> w, w -> c 
			o = in_n * W * H * C + in_w * H * C + in_h * C + in_c;
		}
		else if ((swap_dim_1 == 2 && swap_dim_2 == 3) || (swap_dim_1 == 3 && swap_dim_2 == 2)) { // h -> w, w -> h
			o = in_n * C * W * H + in_c * W * H + in_w * H + in_h;
		}
		y[o] = x[i];
	}

}

Restructure::Restructure(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_restructure_param() == false) << "param.restructure_param() == false";
	auto restructure_param = _param->restructure_param();
	_first_dim = restructure_param.first_dim();
	_second_dim = restructure_param.second_dim();

}

void Restructure::init() {		
	auto in_dim = _inputs[0]->value()->dims();
	auto ou_dim = in_dim;	
	ou_dim[_first_dim] = in_dim[_second_dim];
	ou_dim[_second_dim] = in_dim[_first_dim];
	_outputs[0]->initValue(ou_dim);
	_outputs[0]->initDiff();	
}

void Restructure::forward() {
	auto size = _inputs[0]->value()->size();
	auto dim = _inputs[0]->value()->dims();
	RestructureKernel << < numOfBlocks(size), maxThreadsPerBlock >> > 
		(size, _inputs[0]->value()->gpu_data(), dim[0], dim[1], dim[2], dim[3], _first_dim, _second_dim, _outputs[0]->value()->gpu_data());
	DF_NODE_KERNEL_CHECK();
}

void Restructure::backward() {
	if (_inputs[0]->diff()) {
		auto size = _outputs[0]->diff()->size();
		auto dim = _outputs[0]->diff()->dims();
		RestructureKernel << < numOfBlocks(size), maxThreadsPerBlock >> > 
			(size, _outputs[0]->diff()->gpu_data(), dim[0], dim[1], dim[2], dim[3], _first_dim, _second_dim, _inputs[0]->diff()->gpu_data());
		DF_NODE_KERNEL_CHECK();
	}
}

std::string Restructure::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.resturecture(" + _input_name_for_cpp(0) + ", ";
	cpp += std::to_string(_first_dim) + ", ";
	cpp += std::to_string(_second_dim) + ", ";
	cpp += "\"" + _name + "\");";	
	return cpp;
}
