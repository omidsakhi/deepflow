#pragma once

#include "core/export.h"

#include "core/tensor.h"
#include "generators/mnist_reader.h"

#include "nodes/print.h"
#include "nodes/psnr.h"
#include "nodes/accumulator.h"

#include "core/caffe.h"
#include "nodes/block.h"

#include <string>
#include <memory>
#include <list>

class Session;

class DeepFlowDllExport DeepFlow : public std::enable_shared_from_this<DeepFlow> {
	friend class Session;
	friend class Caffe;
public:
	DeepFlow();

	// GENERATORS
	std::string mnist_reader(std::string folder_path, int batch_size, MNISTReader::MNISTReaderType reader_type, MNISTReader::MNISTOutputType output_type, std::string name = "mnist", std::initializer_list<std::string> phases = {});
	std::string data_generator(std::shared_ptr<deepflow::InitParam> initializer, int num_samples = 1, std::string solver = "", std::string name = "igen", std::initializer_list<std::string> phases = {});
	std::string image_reader(std::string file_path, deepflow::ImageReaderParam_Type type, std::string name = "imread", std::initializer_list<std::string> phases = {});
	std::string image_batch_reader(std::string folder_path, std::initializer_list<int> dims, std::string name = "imbar", std::initializer_list<std::string> phases = {});

	// INITIALIZERS
	std::shared_ptr<deepflow::InitParam> fill(std::initializer_list<int> dims, float value);
	std::shared_ptr<deepflow::InitParam> index_fill(std::initializer_list<int> dims, float offset);
	std::shared_ptr<deepflow::InitParam> zeros(std::initializer_list<int> dims);
	std::shared_ptr<deepflow::InitParam> ones(std::initializer_list<int> dims);
	std::shared_ptr<deepflow::InitParam> random_uniform(std::initializer_list<int> dims, float min, float max);
	std::shared_ptr<deepflow::InitParam> random_normal(std::initializer_list<int> dims, float mean, float stddev);
	std::shared_ptr<deepflow::InitParam> step(std::initializer_list<int> dims, float min, float max);

	// VARIABLES & PLACE HOLDER
	std::string variable(std::shared_ptr<deepflow::InitParam> initializer, std::string solver = "", std::string name = "var", std::initializer_list<std::string> phases = {});
	std::string place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "ph", std::initializer_list<std::string> phases = {});
	
	//CONVOLUTION
	std::string conv2d(std::string input, std::string filter, std::string bias, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string conv2d(std::string input, std::string filter, std::string bias, std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string pooling(std::string input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "maxpool", std::initializer_list<std::string> phases = {});
	std::string transposed_conv2d(std::string input, std::string filter, std::array<int, 4> dims, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name = "tconv", std::initializer_list<std::string> phases = {});

	// MATH
	std::string add(std::string a, std::string b, float alpha, float beta, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string add(std::string a, std::string b, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string square(std::string a, std::string name = "square", std::initializer_list<std::string> phases = {});
	std::string matmul(std::string a, std::string b, std::string name = "ip", std::initializer_list<std::string> phases = {});
	std::string subtract(std::string a, std::string b, std::string name = "sub", std::initializer_list<std::string> phases = {});

	// NEURAL NETS
	std::string bias_add(std::string a, std::string b, std::string name = "bias", std::initializer_list<std::string> phases = {});
	std::string dropout(std::string a, float dropout = 0.5f, bool train_only = true, std::string name = "dropout", std::initializer_list<std::string> phases = {});

	//REDUCTION
	std::string argmax(std::string input, int reduceDimension, std::string name = "argmax", std::initializer_list<std::string> phases = {});
	std::string argmin(std::string input, int reduceDimension, std::string name = "argmin", std::initializer_list<std::string> phases = {});
	std::string reduce_max(std::string input, int reduceDimension, std::string name = "max", std::initializer_list<std::string> phases = {});
	std::string reduce_min(std::string input, int reduceDimension, std::string name = "min", std::initializer_list<std::string> phases = {});
	std::string reduce_mean(std::string input, int reduceDimension, std::string name = "mean", std::initializer_list<std::string> phases = {});
	std::string reduce_sum(std::string input, int reduceDimension, std::string name = "sum", std::initializer_list<std::string> phases = {});
	std::string reduce_absmax(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});
	std::string reduce_norm1(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});
	std::string reduce_norm2(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});

	//ACTIVATIONS
	std::string leaky_relu(std::string a, float negative_slope = 0.01f, std::string name = "leaky_relu", std::initializer_list<std::string> phases = {});
	std::string sigmoid(std::string a, std::string name = "sigmoid", std::initializer_list<std::string> phases = {});
	std::string relu(std::string a, std::string name = "relu", std::initializer_list<std::string> phases = {});
	std::string tanh(std::string a, std::string name = "tanh", std::initializer_list<std::string> phases = {});
	std::string clipped_relu(std::string a, float threshold, std::string name = "clipped_relu", std::initializer_list<std::string> phases = {});
	std::string elu(std::string a, float alpha, std::string name = "elu", std::initializer_list<std::string> phases = {});

	// SELECTORS
	std::string phaseplexer(std::string input_1, std::string phase_1, std::string input_2, std::string phase_2, std::string name = "plex", std::initializer_list<std::string> phases = {});
	std::string random_selector(std::string input_1, std::string input_2, float probability = 0.5f, std::string name = "selector", std::initializer_list<std::string> phases = {});

	// LOSS
	std::shared_ptr<deepflow::NodeParam> softmax_loss(std::string a, std::string b, std::string name = "softmaxloss", std::initializer_list<std::string> phases = {});
	void euclidean_loss(std::string a, std::string b, std::string name = "euclideanloss", std::initializer_list<std::string> phases = {});
		
	// SOLVERS
	std::string sgd_solver(float momentum, float learning_rate, std::string name = "sgd");
	std::string gain_solver(float momentum = 0.99999, float learning_rate = 10e-3f, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f, std::string name = "gain");
	std::string adam_solver(float learning_rate = 10e-2f, float beta1 = 0.99f, float beta2 = 0.999f, float eps = 10e-8f, std::string name = "adam");
	std::string adadelta_solver(float learning_rate = 10e-1f, float momentum = 0.9f, float delta = 1e-6f, std::string name = "adadelta");	
	
	// PRINTERS & DISPLAYS
	void print(std::initializer_list<std::string> inputs, std::string message, Print::PrintTime printTime = Print::END_OF_EPOCH, Print::PrintType = Print::VALUES, std::string name = "print", std::initializer_list<std::string> phases = {});
	void display(std::string input, int delay_msec = 100, deepflow::DisplayParam_DisplayType type = deepflow::DisplayParam_DisplayType_VALUES , std::string name = "disp", std::initializer_list<std::string> phases = {});
	void psnr(std::string a, std::string b,  Psnr::PrintTime printTime = Psnr::END_OF_EPOCH, std::string name = "psnr", std::initializer_list<std::string> phases = {});

	// OTHER
	std::string softmax(std::string a, std::string name = "softmax", std::initializer_list<std::string> phases = {});
	std::string equal(std::string a, std::string b, std::string name = "Equal", std::initializer_list<std::string> phases = {});
	std::string cast_float(std::string input, std::string name = "float", std::initializer_list<std::string> phases = {});
	std::shared_ptr < deepflow::NodeParam > accumulator(std::string input, Accumulator::ResetTime resetTime = Accumulator::ResetTime::EndOfEpoch, std::string name = "acc", std::initializer_list<std::string> phases = {});

	// UTILITIES	
	void define_phase(std::string phase, deepflow::PhaseParam_PhaseBehaviour behaviour);
	void define_train_phase(std::string train_phase);
	void define_validation_phase(std::string validation_phase);
	void define_inference_phase(std::string inference_phase);

	std::shared_ptr<Block> block();
	std::shared_ptr<Session> session();

	// CAFFE
	void load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs);

private:
	std::string _reduce(std::string input, int reduce_dimention, deepflow::ReduceParam_ReduceOp op, deepflow::ReduceParam::OutputType type, int output, std::string name, std::initializer_list<std::string> phases);	

private:
	deepflow::NodeParam _block_node_param;	
	std::shared_ptr<Block> _block;
};