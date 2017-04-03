#include "solvers/gain_solver.h"
#include "core/common_cu.h"
#include "core/variable.h"

#include "core/reader.h"

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

GainSolver::GainSolver(std::shared_ptr<OutputTerminal> loss, SolverParam param) : Solver(loss,param) {
	LOG_IF(FATAL, param.has_gain_solver() == false);
}

void GainSolver::step() {
	const GainSolverParam &param = _param.gain_solver();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveForward();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveBackward();
	int index = 0;	
	for( auto var : _variables)
	{		
		auto output = var->output(0);
		auto size = output->value()->size();
		GainStepKernel << <(size + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock, 0 , _streams[index]>> > (size, (float*) output->value()->mutableData(), (float*) output->diff()->mutableData(), _previous_gradients[index], _gains[index], param.max_gain(), param.min_gain(), param.gain_plus(), param.gain_mult(), param.momentum(), param.learning_rate());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
		index++;
	}

	cudaDeviceSynchronize();	
	
	for (auto node : _variables)
	{
		Variable *var = dynamic_cast<Variable*>(node);
		if (var->snapshot() && _current_iteration % var->snapshotInterval() == 0)
			var->toImage(_current_iteration);
	}

	for (auto reader : _readers)
		reader->nextBatch();
}

void GainSolver::train() {
	LOG(INFO) << "GainSolver::train";
	maxThreadsPerBlock = Node::maxThreadsPerBlock;	
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveNodeFinder<Reader>(_readers);
	LOG(INFO) << _readers.size() << " total readers detected.";
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveInitForward();
	_loss_node->recursiveResetVisit();
	_loss_node->recursiveInitBackward();
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
		GainFillKernel << <(size + maxThreadsPerBlock - 1) / maxThreadsPerBlock, maxThreadsPerBlock >> >(size, _gains.back());
		LOG_IF(FATAL, cudaPeekAtLastError() != 0);		
	}	
	for (int i = 0; i < _param.max_iteration(); ++i)
	{
		_current_iteration = i;
		step();
	}
}