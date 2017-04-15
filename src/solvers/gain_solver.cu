#include "solvers/gain_solver.h"
#include "core/common_cu.h"
#include "core/variable.h"

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

GainSolver::GainSolver(const SolverParam &param): Solver(param) {
	LOG_IF(FATAL, param.has_gain_solver() == false) << "param.has_gain_solver() == false";
	_my_param = param.gain_solver();
}

void GainSolver::apply(std::shared_ptr<Variable> var) {
	if (_initialized == false) 
		init(var);
	auto output = var->output(0);
	auto size = output->value()->size();
	GainStepKernel << <numOfBlocks(size), maxThreadsPerBlock>> > (size, (float*) output->value()->mutableData(), (float*) output->diff()->data(), _previous_gradient, _gain, _my_param.max_gain(), _my_param.min_gain(), _my_param.gain_plus(), _my_param.gain_mult(), _my_param.momentum(), _my_param.learning_rate());
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
}

void GainSolver::init(std::shared_ptr<Variable> var) {
	auto size = var->output(0)->value()->size();
	auto sizeInBytes = var->output(0)->value()->sizeInBytes();		
	LOG_IF(FATAL, cudaMalloc(&_previous_gradient, sizeInBytes) != 0);
	LOG_IF(FATAL, cudaMemset(_previous_gradient, 0, sizeInBytes) != 0);
	LOG_IF(FATAL, cudaMalloc(&_gain, sizeInBytes) != 0);
	GainFillKernel <<<numOfBlocks(size), maxThreadsPerBlock>>>(size, _gain);
	LOG_IF(FATAL, cudaPeekAtLastError() != 0);	
	_initialized = true;
}