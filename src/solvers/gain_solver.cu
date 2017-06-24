#include "solvers/gain_solver.h"
#include "core/common_cu.h"
#include "nodes/variable.h"

#include <glog/logging.h>

__global__
void GainFillKernel(int n, float *a)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = 1.0f;
}

__global__
void GainStepKernel(const int n, float *w, const float *g, float *m, float *gain, const float max_gain, const float min_gain, const float gain_plus, const float gain_mult, const float momentum, const float learning_rate)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float _gain = gain[i];
		float _g = g[i];		
		if (_g * m[i] > 0)
			_gain += gain_plus;
		else
			_gain *= gain_mult;
		if (_gain > max_gain)
			_gain = max_gain;
		if (_gain < min_gain)
			_gain = min_gain;
		gain[i] = _gain;
		m[i] = _g;
		w[i] = momentum * w[i] + learning_rate * _g * _gain;
	}
}

GainSolver::GainSolver(deepflow::SolverParam *param): Solver(param) {
	LOG_IF(FATAL, param->has_gain_solver() == false) << "param.has_gain_solver() == false";
	_my_param = param->mutable_gain_solver();
}

void GainSolver::apply(std::shared_ptr<Variable> var) {
	auto context = var->executionContext();
	bool verbos = (context && context->debug_level > 3) ? true : false;
	if (_initialized == false) {
		LOG_IF(INFO, verbos) << "SOLVER " << name() << " FOR VARIABLE " << var->name();
		init(var);
	}
	if (_enable_input) {
		float value = _enable_input->value()->toFloat();
		bool is_enable = value >= 1;
		if (!is_enable) {
			LOG_IF(INFO, verbos) << "SOLVER " << name() << " **NOT** APPLIED ON " << var->name();
			return;
		}
	}
	LOG_IF(INFO, verbos) << "APPLYING SOLVER " << name() << " ON " << var->name();
	auto output = var->output(0);
	auto size = output->value()->size();	
	GainStepKernel << <numOfBlocks(size), maxThreadsPerBlock>> > (size, (float*) output->value()->mutableData(), (float*) output->diff()->data(), _previous_gradient, _gain, _my_param->max_gain(), _my_param->min_gain(), _my_param->gain_plus(), _my_param->gain_mult(), _my_param->momentum(), _my_param->learning_rate());
	DF_KERNEL_CHECK();
}

void GainSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();		
	DF_CUDA_CHECK(cudaMalloc(&_previous_gradient, sizeInBytes));
	DF_CUDA_CHECK(cudaMemset(_previous_gradient, 0, sizeInBytes));
	DF_CUDA_CHECK(cudaMalloc(&_gain, sizeInBytes));	
	GainFillKernel <<<numOfBlocks(size), maxThreadsPerBlock>>>(size, _gain);
	DF_KERNEL_CHECK();
	_initialized = true;
}

std::string GainSolver::to_cpp() const
{	
	std::string cpp = "auto " + name() + " = df.gain_solver(";
	cpp += std::to_string(_my_param->momentum()) + ", ";
	cpp += std::to_string(_my_param->learning_rate()) + ", ";
	cpp += std::to_string(_my_param->max_gain()) + ", ";
	cpp += std::to_string(_my_param->min_gain()) + ", ";
	cpp += std::to_string(_my_param->gain_plus()) + ", ";
	cpp += std::to_string(_my_param->gain_mult()) + ", ";
	cpp += "\"" + name() + "\"";
	cpp += ");";
	return cpp;
}
