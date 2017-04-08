#include "solvers/gain_solver.h"
#include "core/common_cu.h"
#include "core/variable.h"

#include "core/reader.h"
#include "observers/forward.h"
#include "observers/backward.h"
#include "observers/reset.h"

#include <glog/logging.h>

__global__
void GainFillKernel(int n, float *a)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) a[i] = 1.0f;
}

__global__
void GainStepKernel(const int n, float *current_weight, float *current_gradient, float *previous_gradient, float *gain, const float max_gain, const float min_gain, const float gain_plus, const float gain_mult, const float momentum, const float learning_rate)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		if (previous_gradient[i] * current_gradient[i] > 0)
			gain[i] += gain_plus;
		else
			gain[i] *= gain_mult;
		if (gain[i] > max_gain)
			gain[i] = max_gain;
		if (gain[i] < min_gain)
			gain[i] = min_gain;
		previous_gradient[i] = current_gradient[i];
		current_gradient[i] *= gain[i];
		current_weight[i] = momentum * current_weight[i] + learning_rate * current_gradient[i];
	}
}

GainSolver::GainSolver(NodeOutputPtr loss, const SolverParam &param) : Solver(loss,param) {
	LOG_IF(FATAL, param.has_gain_solver() == false) << "param.has_gain_solver() == false";
	_my_param = param.gain_solver();
}

void GainSolver::train_step() {
	if (_initialized == false) {
		init();
	}	
	ResetObserver resetObserver;
	ForwardObserver forwardObserver;
	BackwardObserver backwardObserver;
	_loss_node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	_loss_node->traverse(&forwardObserver, TraverseOrder::PostOrder, false);
	_loss_node->traverse(&resetObserver, TraverseOrder::PreOrder, true);
	_loss_node->traverse(&backwardObserver, TraverseOrder::PreOrder, false);	
	int index = 0;	
	for( auto var : _variables)
	{		
		auto output = var->output(0);
		auto size = output->value()->size();
		GainStepKernel << <numOfBlocks(size), maxThreadsPerBlock, 0 , _streams[index]>> > (size, (float*) output->value()->mutableData(), (float*) output->diff()->mutableData(), _previous_gradients[index], _gains[index], _my_param.max_gain(), _my_param.min_gain(), _my_param.gain_plus(), _my_param.gain_mult(), _my_param.momentum(), _my_param.learning_rate());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
		index++;
	}

	cudaDeviceSynchronize();
	
	for (auto var : _variables)
	{		
		if (var->snapshot() && _current_step % var->snapshotInterval() == 0)
			var->toImage(_current_step);
	}

	_current_step++;
}

void GainSolver::init() {
	for (auto var : _variables)
	{
		_streams.push_back(cudaStream_t());
		cudaStreamCreate(&_streams.back());
		auto size = var->output(0)->value()->size();
		auto sizeInBytes = var->output(0)->value()->sizeInBytes();
		_previous_gradients.push_back(NULL);
		LOG_IF(FATAL, cudaMalloc(&_previous_gradients.back(), sizeInBytes) != 0);
		LOG_IF(FATAL, cudaMemset(_previous_gradients.back(), 0, sizeInBytes) != 0);
		_gains.push_back(NULL);
		LOG_IF(FATAL, cudaMalloc(&_gains.back(), sizeof(float) * size) != 0);
		GainFillKernel <<<numOfBlocks(size), maxThreadsPerBlock>>>(size, _gains.back());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);
	}
	_initialized = true;
}