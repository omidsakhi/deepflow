
#include "core/common_cu.h"

#include "nodes/euclidean_distance.h"

__global__
void EuclideanDistanceForwardKernel(const int n, const float * __restrict__ x1, const float * __restrict__ x2, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		float tmp = x2[i] - x1[i];
		y[i] = tmp * tmp;
	}
}

__global__
void EuclideanDistanceBackwardKernel(const int n, const float * __restrict__ x1, const float * __restrict__ x2, const float * __restrict__ d, float * __restrict__ y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		y[i] += 2 * (x2[i] - x1[i]) * d[i];
}

EuclideanDistance::EuclideanDistance(deepflow::NodeParam *param) : Node(param) {
	LOG_IF(FATAL, param->has_euclidean_distance_param() == false) << "param.has_euclidean_distance_param() == false";
}

void EuclideanDistance::initForward() {	
	LOG(INFO) << "Initializing Euclidean Distance " << _name << " - " << _inputs[0]->value()->shape();
	LOG_IF(FATAL, _inputs[0]->value()->size() != _inputs[1]->value()->size()) << "Input " << _inputs[0]->value()->shape() << " != " << " Target " << _inputs[1]->value()->shape();	
	_outputs[0]->initValue(_inputs[0]->dims());
}

void EuclideanDistance::initBackward() {
	_outputs[0]->initDiff();
}

void EuclideanDistance::forward() {
	auto size = _inputs[0]->value()->size();
	EuclideanDistanceForwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data(), (float*)_outputs[0]->value()->mutableData());
	DF_KERNEL_CHECK();
}

void EuclideanDistance::backward() {		
	auto size = _inputs[0]->value()->size();
	if (_inputs[0]->connectedNode()->propagateBack()) {
		EuclideanDistanceBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[0]->value()->data(), (float*)_inputs[1]->value()->data() , (float*)_outputs[0]->diff()->data(), (float*)_inputs[0]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
	if (_inputs[1]->connectedNode()->propagateBack()) {
		EuclideanDistanceBackwardKernel << < numOfBlocks(size), maxThreadsPerBlock >> > (size, (float*)_inputs[1]->value()->data(), (float*)_inputs[0]->value()->data() , (float*)_outputs[0]->diff()->data(), (float*)_inputs[1]->diff()->mutableData());
		DF_KERNEL_CHECK();
	}
}

std::string EuclideanDistance::to_cpp() const
{
	std::string cpp = "auto " + _name + " = df.euclidean_distance(" + _input_name_for_cpp(0) + ", " + _input_name_for_cpp(1) + ", ";
	cpp += "\"" + _name + "\", ";
	cpp += "{" + _to_cpp_phases() + "});";
	return cpp;
}
