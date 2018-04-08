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
	enum ImreadImageType {
		IMREAD_GRAY,
		IMREAD_COLOR
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
	std::string mnist_reader(std::string folder_path, int batch_size, MNISTReader::MNISTReaderType reader_type, MNISTReader::MNISTOutputType output_type, std::string name = "mnist");
	std::string data_generator(std::string initializer, std::string solver = "", std::string name = "igen");
	std::string imread(std::string file_path, ImreadImageType type, std::string name = "imread");
	std::string image_batch_reader(std::string folder_path, std::initializer_list<int> dims, bool randomize = false, std::string name = "imbar");

	// INITIALIZERS
	std::string fill(std::initializer_list<int> dims, float value, std::string name = "fill");
	std::string index_fill(std::initializer_list<int> dims, float offset, std::string name = "index_fill");
	std::string zeros(std::initializer_list<int> dims, std::string name = "zero");
	std::string ones(std::initializer_list<int> dims, std::string = "ones");
	std::string random_uniform(std::initializer_list<int> dims, float min, float max, std::string name = "random_uniform");
	std::string random_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name = "random_normal");
	std::string truncated_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name = "truncated_normal");
	std::string step(std::initializer_list<int> dims, float min, float max, std::string name = "step");
	std::string three_state(std::initializer_list<int> dims, std::string name = "three_state");
	std::string gradient(std::initializer_list<int> dims, std::string name = "gradient");

	// VARIABLES & PLACE HOLDER
	std::string variable(std::string initializer, std::string solver = "", std::string name = "var");
	std::string place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "ph");		

	//CONVOLUTION
	std::string conv2d(std::string input, std::string filter, int pad_top_bottom = 0, int pad_left_right = 0, int vertical_filter_stride = 1, int horizontal_filter_stride = 1, int filter_height_dilation = 1, int filter_width_dialation = 1, std::string name = "conv");	
	std::string conv2d(std::string input, int input_channels, int output_channels, int kernel, int pad, int stride, bool has_bias, std::string solver, std::string name);	
	std::string transposed_conv2d(std::string input, int input_channels, int output_channels, int kernel, int pad, int stride, bool has_bias, std::string solver, std::string name);	
	std::string transposed_conv2d(std::string input, std::string filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name = "tconv");

	// RESTRUCTURE
	std::string restructure(std::string input, int first_dims, int second_dim, std::string name = "restructure");
	std::string pooling(std::string input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "maxpool");
	std::string lifting(std::string input, LiftingMode mode, std::string name = "lifting");
	std::string upsample(std::string input, std::string name = "upsample");
	std::string patching(std::string input, PatchingMode mode, int num_vertical_patches, int num_horizontal_patches, std::string name = "patching");
	std::string patch_sampling(std::string input, int patch_width, int patch_height, std::string name = "patch_sampling");
	std::string resize(std::string input, float height_scale, float width_scale, std::string name = "resize");
	std::string concate(std::string input1, std::string input2, std::string name = "concate");
	std::string reshape(std::string input, std::array<int, 4> output_dims, std::string name = "reshape");	

	// MATH
	std::string add(std::string a, std::string b, float alpha, float beta, std::string name = "add");
	std::string add(std::string a, std::string b, std::string name = "add");
	std::string dot(std::string a, std::string b, std::string name = "dot");
	std::string square(std::string a, std::string name = "square");
	std::string abs(std::string a, std::string name = "abs");
	std::string exp(std::string a, std::string name = "exp");
	std::string log(std::string a, float coef = 1.0f, std::string name = "log");	
	std::string subtract(std::string a, std::string b, std::string name = "sub");
	std::string square_error(std::string a, std::string b, std::string name = "SquareError");
	std::string softmax(std::string a, std::string name = "softmax");

	// matmul
	std::string matmul(std::string a, std::string b, std::string name = "ip");
	std::string dense(std::string input, std::initializer_list<int> dims, bool has_bias, std::string solver, std::string name);	

	// NEURAL NETS
	std::string bias_add(std::string input, std::string weights, std::string name = "bias");
	std::string bias_add(std::string input, int output_channels, std::string solver, std::string name);
	std::string dropout(std::string a, float dropout = 0.5f, bool train_only = true, std::string name = "dropout");

	//REDUCTION
	
	std::string argmax(std::string input, int reduceDimension, std::string name = "argmax");
	std::string argmin(std::string input, int reduceDimension, std::string name = "argmin");
	std::string reduce_max(std::string input, int reduceDimension, std::string name = "max");
	std::string reduce_min(std::string input, int reduceDimension, std::string name = "min");
	std::string reduce_mean(std::string input, int reduceDimension, std::string name = "mean");
	std::string reduce_sum(std::string input, int reduceDimension, std::string name = "sum");
	std::string reduce_absmax(std::string input, int reduceDimension, std::string name = "absmax");
	std::string reduce_norm1(std::string input, int reduceDimension, std::string name = "absmax");
	std::string reduce_norm2(std::string input, int reduceDimension, std::string name = "absmax");
	std::string reduce_mean(std::string input, std::string name = "mean");
	std::string reduce_sum(std::string input, std::string name = "mean");
	

	//ACTIVATIONS
	std::string leaky_relu(std::string a, float negative_slope = 0.01f, std::string name = "leaky_relu");
	std::string sigmoid(std::string a, std::string name = "sigmoid");
	std::string relu(std::string a, std::string name = "relu");
	std::string prelu(std::string input, std::string w, std::string name = "prelu");
	std::string prelu(std::string input, int output_channels, std::string solver, std::string name);
	std::string dprelu(std::string input, std::string a, std::string name = "dprelu");
	std::string tanh(std::string a, std::string name = "tanh");
	std::string clipped_relu(std::string a, float threshold, std::string name = "clipped_relu");
	std::string elu(std::string a, float alpha, std::string name = "elu");

	// SELECTORS	
	std::string random_selector(std::string input_1, std::string input_2, float probability = 0.5f, std::string name = "selector");
	std::string multiplexer(std::initializer_list<std::string> inputs, std::string name = "multiplexer");
	std::string switcher(std::string input, std::string name = "switch");
				
	// SOLVERS
	std::string sgd_solver(float momentum, float learning_rate, std::string name = "sgd");
	std::string gain_solver(float momentum = 0.98f, float learning_rate = 10e-2f, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f, std::string name = "gain");
	std::string adam_solver(float learning_rate = 10e-2f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 10e-8f, std::string name = "adam");
	std::string adadelta_solver(float learning_rate = 10e-2f, float momentum = 0.5f, float delta = 1e-6f, std::string name = "adadelta");
	std::string rmsprop_solver(float learning_rate = 10e-2f, float rms_decay = 0.9f, float eps = 0.001f, std::string name = "rmsprop");
	
	// PRINTERS & DISPLAYS
	void print(std::initializer_list<std::string> inputs, std::string message, ActionType = ActionType::VALUES, std::string name = "print");
	void logger(std::initializer_list<std::string> inputs, std::string file_path, std::string message, ActionType loggingType = ActionType::VALUES, std::string name = "logger");
	std::string display(std::string input, int delay_msec = 100, ActionType dislayType = ActionType::VALUES, int epoch_frequency = 1, bool draw_iteration = false, std::string name = "disp");
	std::string imwrite(std::string input, std::string filename, bool draw_iteration = false, std::string name = "disp");
	void psnr(std::string a, std::string b, std::string name = "psnr");	

	// NORMALIZATION
	std::string batch_normalization(std::string input, std::string scale, std::string bias, NormalizationMode mode = DeepFlow::SPATIAL, float exp_factor = 0.9f, bool cache = true, std::string name = "batch_norm");
	std::string batch_normalization(std::string input, std::string solver, int output_channels, float exp_factor, std::string name);
	std::string lrn(std::string input, std::string name = "lrn", int n = 5, float alpha = 1e-4f, float beta = 0.75f, float k = 2.0f);	

	// SOCKET IO
	void sio_output(std::initializer_list<std::string> inputs, std::string host = "127.0.0.1", int port = 6643, std::string name = "sio_output");

	// OTHER	
	std::string loss(std::string a, ReduceOp op, float alpha = 1.0f, float beta = 0.0f, std::string name = "loss");
	std::string equal(std::string a, std::string b, std::string name = "Equal");
	std::string cast_float(std::string input, std::string name = "float");	
	std::array<std::string,2> accumulator(std::string input, std::string name = "acc");
	std::string replay_memory(std::string input, int capacity, std::string name = "replay_memory");
	std::string batch_stddev(std::string input, std::string name = "batch_stddev");
	std::string pass_through(std::string input, bool stop_gradients, std::string name = "pass_through");
	std::string gaussian(std::string mean, std::string sigma, std::string name = "gaussian");
	std::string gaussian_kernel(int window_size, float sigma, std::string name = "gaussian_weights");
	std::string gaussian_blur(std::string input, int window_size, float sigma, std::string name = "gaussian_blur");
	std::string identity(std::string input, float scale, int input_channels, int output_channels, std::string solver, std::string name = "identity");

	std::shared_ptr<Block> block();
	std::shared_ptr<Session> session();

	// CAFFE
	void load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose = false);

private:
	std::string _reduce(std::string input, int reduce_dimention, deepflow::ReduceParam_ReduceOp op, deepflow::ReduceParam::OutputType type, int output, std::string name);	
	std::string _reduce_all(std::string input, ReduceAllOp op, std::string name);


private:		
	std::shared_ptr<Block> _block;
};