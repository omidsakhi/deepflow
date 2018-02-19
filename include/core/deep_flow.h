#pragma once

#include "core/export.h"

#include "core/tensor.h"
#include "generators/mnist_reader.h"

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
	enum PhaseBehaviour {
		TRAIN_AND_INFERENCE,
		TRAIN,
		VALIDATION,
		INFERENCE
	};
	enum ActionTime {
		EVERY_PASS = 0,
		END_OF_EPOCH = 1,
		NEVER = 2
	};
	enum ActionType {
		VALUES = 0,
		DIFFS = 1,
		VALUES_AND_DIFFS = 2
	};
	enum NormalizationMode {
		PER_ACTIVATION = 0,
		SPATIAL = 1
	};

	enum ReduceOp {
		SUM = 0,
		AVG = 5,
	};

	enum LiftingMode {
		LIFT_UP_REGULAR = 0,
		LIFT_DOWN_REGULAR = 1,
		LIFT_UP_FLIP = 2,
		LIFT_DOWN_FLIP = 3
	};

	enum PatchingMode {
		PATCHING_UPSAMPLES = 0,
		PATCHING_DOWNSAMPLES = 1,
		PATCHING_UPCHANNELS = 2,
		PATCHING_DOWNCHANNELS = 3
	};

	enum ReduceAllOp {
		REDUCE_ALL_SUM,
		REDUCE_ALL_AVG
	};

	DeepFlow();
	DeepFlow(std::shared_ptr<Block> block);

	// GENERATORS
	std::string mnist_reader(std::string folder_path, int batch_size, MNISTReader::MNISTReaderType reader_type, MNISTReader::MNISTOutputType output_type, std::string name = "mnist", std::initializer_list<std::string> phases = {});
	std::string data_generator(std::string initializer, int num_samples = 1, std::string solver = "", std::string name = "igen", std::initializer_list<std::string> phases = {});
	std::string image_reader(std::string file_path, deepflow::ImageReaderParam_Type type, std::string name = "imread", std::initializer_list<std::string> phases = {});
	std::string image_batch_reader(std::string folder_path, std::initializer_list<int> dims, bool randomize = false, std::string name = "imbar", std::initializer_list<std::string> phases = {});

	// INITIALIZERS
	std::string fill(std::initializer_list<int> dims, float value, std::string name = "fill");
	std::string index_fill(std::initializer_list<int> dims, float offset, std::string name = "index_fill");
	std::string zeros(std::initializer_list<int> dims, std::string name = "zero");
	std::string ones(std::initializer_list<int> dims, std::string = "ones");
	std::string random_uniform(std::initializer_list<int> dims, float min, float max, std::string name = "random_uniform");
	std::string random_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name = "random_normal");
	std::string step(std::initializer_list<int> dims, float min, float max, std::string name = "step");
	std::string three_state(std::initializer_list<int> dims, std::string name = "three_state");

	// VARIABLES & PLACE HOLDER
	std::string variable(std::string initializer, std::string solver = "", std::string name = "var", std::initializer_list<std::string> phases = {});
	std::string place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "ph", std::initializer_list<std::string> phases = {});		

	//CONVOLUTION
	std::string conv2d(std::string input, std::string filter, std::string bias, float negative_slope /* NOT SUPPORTED YET BY CUDNN */, int pad_top_bottom = 0, int pad_left_right = 0, int vertical_filter_stride = 1, int horizontal_filter_stride = 1, int filter_height_dilation = 1, int filter_width_dialation = 1, std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string conv2d(std::string input, std::string filter, int pad_top_bottom = 0, int pad_left_right = 0, int vertical_filter_stride = 1, int horizontal_filter_stride = 1, int filter_height_dilation = 1, int filter_width_dialation = 1, std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string transposed_conv2d(std::string input, std::string filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name = "tconv", std::initializer_list<std::string> phases = {});

	// RESTRUCTURE
	std::string restructure(std::string input, int first_dims, int second_dim, std::string name = "restructure", std::initializer_list<std::string> phases = {});
	std::string pooling(std::string input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "maxpool", std::initializer_list<std::string> phases = {});
	std::string lifting(std::string input, LiftingMode mode, std::string name = "lifting", std::initializer_list<std::string> phases = {});
	std::string upsample(std::string input, std::string name = "upsample", std::initializer_list<std::string> phases = {});
	std::string patching(std::string input, PatchingMode mode, int num_vertical_patches, int num_horizontal_patches, std::string name = "patching", std::initializer_list<std::string> phases = {});
	std::string resize(std::string input, float height_scale, float width_scale, std::string name = "resize", std::initializer_list<std::string> phases = {});

	// MATH
	std::string add(std::string a, std::string b, float alpha, float beta, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string add(std::string a, std::string b, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string dot(std::string a, std::string b, std::string name = "dot", std::initializer_list<std::string> phases = {});
	std::string square(std::string a, std::string name = "square", std::initializer_list<std::string> phases = {});
	std::string abs(std::string a, std::string name = "abs", std::initializer_list<std::string> phases = {});
	std::string exp(std::string a, std::string name = "exp", std::initializer_list<std::string> phases = {});
	std::string log(std::string a, float coef = 1.0f, std::string name = "log", std::initializer_list<std::string> phases = {});
	std::string matmul(std::string a, std::string b, std::string name = "ip", std::initializer_list<std::string> phases = {});
	std::string subtract(std::string a, std::string b, std::string name = "sub", std::initializer_list<std::string> phases = {});
	std::string square_error(std::string a, std::string b, std::string name = "SquareError", std::initializer_list<std::string> phases = {});
	std::string softmax(std::string a, std::string name = "softmax", std::initializer_list<std::string> phases = {});

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
	std::string reduce_mean(std::string input, std::string name = "mean", std::initializer_list<std::string> phases = {});
	std::string reduce_sum(std::string input, std::string name = "mean", std::initializer_list<std::string> phases = {});
	

	//ACTIVATIONS
	std::string leaky_relu(std::string a, float negative_slope = 0.01f, std::string name = "leaky_relu", std::initializer_list<std::string> phases = {});
	std::string sigmoid(std::string a, std::string name = "sigmoid", std::initializer_list<std::string> phases = {});
	std::string relu(std::string a, std::string name = "relu", std::initializer_list<std::string> phases = {});
	std::string prelu(std::string input, std::string w, std::string name = "prelu", std::initializer_list<std::string> phases = {});
	std::string tanh(std::string a, std::string name = "tanh", std::initializer_list<std::string> phases = {});
	std::string clipped_relu(std::string a, float threshold, std::string name = "clipped_relu", std::initializer_list<std::string> phases = {});
	std::string elu(std::string a, float alpha, std::string name = "elu", std::initializer_list<std::string> phases = {});

	// SELECTORS
	std::string phaseplexer(std::string input_1, std::string phase_1, std::string input_2, std::string phase_2, std::string name = "plex", std::initializer_list<std::string> phases = {});
	std::string random_selector(std::string input_1, std::string input_2, float probability = 0.5f, std::string name = "selector", std::initializer_list<std::string> phases = {});
	std::string multiplexer(std::initializer_list<std::string> inputs, std::string name = "multiplexer", std::initializer_list<std::string> phases = {});
	std::string switcher(std::string input, std::string name = "switch", std::initializer_list<std::string> phases = {});
				
	// SOLVERS
	std::string sgd_solver(float momentum, float learning_rate, std::string name = "sgd", std::string enable_input = "");
	std::string gain_solver(float momentum = 0.98f, float learning_rate = 10e-2f, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f, std::string name = "gain", std::string enable_input = "");
	std::string adam_solver(float learning_rate = 10e-2f, float beta1 = 0.5f, float beta2 = 0.999f, float eps = 10e-8f, std::string name = "adam",std::string enable_input = "");
	std::string adadelta_solver(float learning_rate = 10e-2f, float momentum = 0.5f, float delta = 1e-6f, std::string name = "adadelta", std::string enable_input = "");
	
	// PRINTERS & DISPLAYS
	void print(std::initializer_list<std::string> inputs, std::string message, ActionTime printTime = ActionTime::END_OF_EPOCH, ActionType = ActionType::VALUES, std::string name = "print", std::initializer_list<std::string> phases = {});
	void logger(std::initializer_list<std::string> inputs, std::string file_path, std::string message, ActionTime loggingTime = ActionTime::END_OF_EPOCH, ActionType loggingType = ActionType::VALUES, std::string name = "logger", std::initializer_list<std::string> phases = {});
	std::string display(std::string input, int delay_msec = 100, ActionTime displayTime = ActionTime::EVERY_PASS, ActionType dislayType = ActionType::VALUES, int epoch_frequency = 1, std::string name = "disp", std::initializer_list<std::string> phases = {});
	std::string imwrite(std::string input, std::string filename, std::string name = "disp", std::initializer_list<std::string> phases = {});
	void psnr(std::string a, std::string b, ActionTime printTime = ActionTime::END_OF_EPOCH, std::string name = "psnr", std::initializer_list<std::string> phases = {});	

	// NORMALIZATION
	std::string batch_normalization(std::string input, std::string scale, std::string bias, NormalizationMode mode = DeepFlow::SPATIAL, bool cache = true, std::string name = "batch_norm", std::initializer_list<std::string> phases = {});
	std::string lrn(std::string input, std::string name = "lrn", int n = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 2.0f, std::initializer_list<std::string> phases = {});

	// SOCKET IO
	void sio_output(std::initializer_list<std::string> inputs, ActionTime printTime = ActionTime::EVERY_PASS, std::string host = "127.0.0.1", int port = 6643, std::string name = "sio_output", std::initializer_list<std::string> phases = {});

	// OTHER	
	std::string loss(std::string a, std::string coef, ReduceOp op, std::string name = "loss", std::initializer_list<std::string> phases = {});
	std::string equal(std::string a, std::string b, std::string name = "Equal", std::initializer_list<std::string> phases = {});
	std::string cast_float(std::string input, std::string name = "float", std::initializer_list<std::string> phases = {});	
	std::array<std::string,2> accumulator(std::string input, ActionTime resetTime = ActionTime::END_OF_EPOCH, std::string name = "acc", std::initializer_list<std::string> phases = {});
	std::string replay_memory(std::string input, int capacity, std::string name = "replay_memory", std::initializer_list<std::string> phases = {});

	// UTILITIES	
	std::string define_phase(std::string phase, PhaseBehaviour behaviour = TRAIN_AND_INFERENCE);
	std::string define_train_phase(std::string train_phase);
	std::string define_validation_phase(std::string validation_phase);
	std::string define_inference_phase(std::string inference_phase);

	std::shared_ptr<Block> block();
	std::shared_ptr<Session> session();

	// CAFFE
	void load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose = false);

private:
	std::string _reduce(std::string input, int reduce_dimention, deepflow::ReduceParam_ReduceOp op, deepflow::ReduceParam::OutputType type, int output, std::string name, std::initializer_list<std::string> phases);	
	std::string _reduce_all(std::string input, ReduceAllOp op, std::string name, std::initializer_list<std::string> phases);


private:		
	std::shared_ptr<Block> _block;
};