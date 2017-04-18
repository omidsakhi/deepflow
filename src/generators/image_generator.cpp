#include "generators/image_generator.h"
#include "core/variable.h"
#include "core/initializer.h"
#include "core/common_cu.h"

ImageGenerator::ImageGenerator(std::shared_ptr<Initializer> initializer, const NodeParam &param) : Generator(param), Variable(initializer,param) {
	LOG_IF(FATAL, param.generator_param().has_image_generator_param() == false) << "param.generator_param().has_image_generator_param() == false";
}

void ImageGenerator::nextBatch() {

}

void ImageGenerator::initForward() {
	_initializer->init();
	_outputs[0]->initValue(_initializer->dims());
	LOG(INFO) << "Initializing Variable " << _name << " - " << _outputs[0]->value()->shape();
	if (_param.variable_param().has_weights()) {
		DF_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _param.variable_param().weights().weight().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else if (_initializer->param().has_init_data()) {
		DF_CUDA_CHECK(cudaMemcpy(_outputs[0]->value()->mutableData(), _initializer->param().init_data().weight().data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyHostToDevice));
	}
	else {
		_initializer->apply(this);
		for (int i = 0; i < _outputs[0]->value()->size(); ++i)
			_param.mutable_variable_param()->mutable_init_param()->mutable_init_data()->add_weight(0);
		DF_CUDA_CHECK(cudaMemcpy(_param.mutable_variable_param()->mutable_init_param()->mutable_init_data()->mutable_weight()->mutable_data(), _outputs[0]->value()->data(), _outputs[0]->value()->sizeInBytes(), cudaMemcpyDeviceToHost));
	}
}

void ImageGenerator::initBackward() {
	_outputs[0]->initDiff();
}

void ImageGenerator::forward() {

}

void ImageGenerator::backward() {

}

bool ImageGenerator::isLastBatch() {
	return false;
}