#pragma once

#include "core/export.h"

#include "core/tensor.h"
#include "readers/mnist_reader.h"
#include "ops/print.h"
#include "ops/accumulator.h"

#include <string>
#include <memory>
#include <list>

class Session;

class DeepFlowDllExport DeepFlow {	
	friend class Session;
public:
	// READERS
	std::shared_ptr<NodeParam> mnist_reader(std::string folder_path, int batch_size, MNISTReaderType type, std::string name = "mnist", std::initializer_list<std::string> phases = {});

	// INITIALIZERS
	std::shared_ptr<InitParam> fill(std::initializer_list<int> dims, float value, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<InitParam> index_fill(std::initializer_list<int> dims, float offset, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<InitParam> zeros(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<InitParam> ones(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<InitParam> random_uniform(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<InitParam> step(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);

	// VARIABLES & PLACE HOLDERS
	std::string variable(std::shared_ptr<InitParam> initializer, std::shared_ptr<SolverParam> solver, std::string name = "var", std::initializer_list<std::string> phases = {});	
	std::string place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "ph", std::initializer_list<std::string> phases = {});

	// OPS
	std::string add(std::string a, std::string b, float alpha, float beta, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string add(std::string a, std::string b, std::string name = "add", std::initializer_list<std::string> phases = {});
	std::string bias_add(std::string a, std::string b, std::string name = "bias", std::initializer_list<std::string> phases = {});
	std::string subtract(std::string a, std::string b, std::string name = "sub", std::initializer_list<std::string> phases = {});
	std::string softmax(std::string a, std::string name = "softmax", std::initializer_list<std::string> phases = {});
	std::string square(std::string a, std::string name = "square", std::initializer_list<std::string> phases = {});
	std::string matmul(std::string a, std::string b, std::string name = "ip", std::initializer_list<std::string> phases = {});
	std::string relu(std::string a, float negative_slope = 0.01f, std::string name = "relu", std::initializer_list<std::string> phases = {});
	std::string dropout(std::string a, float dropout = 0.5f, std::string name = "dropout", std::initializer_list<std::string> phases = {});
	std::string conv2d(std::string input, std::string filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation , std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string conv2d(std::string input, std::string filter, std::string name = "conv", std::initializer_list<std::string> phases = {});
	std::string pooling(std::string input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "maxpool", std::initializer_list<std::string> phases = {});
	std::string argmax(std::string input, int reduceDimension, std::string name = "argmax", std::initializer_list<std::string> phases = {});
	std::string argmin(std::string input, int reduceDimension, std::string name = "argmin", std::initializer_list<std::string> phases = {});
	std::string reduce_max(std::string input, int reduceDimension, std::string name = "max", std::initializer_list<std::string> phases = {});	
	std::string reduce_min(std::string input, int reduceDimension, std::string name = "min", std::initializer_list<std::string> phases = {});
	std::string reduce_mean(std::string input, int reduceDimension, std::string name = "mean", std::initializer_list<std::string> phases = {});	
	std::string reduce_sum(std::string input, int reduceDimension, std::string name = "sum", std::initializer_list<std::string> phases = {});
	std::string reduce_absmax(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});
	std::string reduce_norm1(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});
	std::string reduce_norm2(std::string input, int reduceDimension, std::string name = "absmax", std::initializer_list<std::string> phases = {});
	std::string equal(std::string a, std::string b, std::string name = "Equal", std::initializer_list<std::string> phases = {});
	std::string cast_float(std::string input, std::string name = "float", std::initializer_list<std::string> phases = {});
	std::string accumulator(std::string input, Accumulator::ResetTime resetTime = Accumulator::ResetTime::EndOfEpoch, std::string name = "acc", std::initializer_list<std::string> phases = {});
	void display(std::string input, int delay_msec = 100, std::string name = "disp", std::initializer_list<std::string> phases = {});

	// PHASE
	std::string phaseplexer(std::string input_1, std::string phase_1, std::string input_2, std::string phase_2, std::string name = "plex", std::initializer_list<std::string> phases = {});

	// LOSS
	std::shared_ptr<NodeParam> softmax_loss(std::string a, std::string b, std::string name = "softmaxloss", std::initializer_list<std::string> phases = {});
		
	// SOLVERS
	std::shared_ptr<SolverParam> sgd_solver(float momentum, float learning_rate, std::string name = "sgd");
	std::shared_ptr<SolverParam> gain_solver(float momentum = 0.99999, float learning_rate = 10E-3f, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f, std::string name = "gain");
	std::shared_ptr<SolverParam> adam_solver(float learning_rate = 10E-2f, float beta1 = 0.99f, float beta2 = 0.999f, float eps = 10E-8f, std::string name = "adam");
	std::shared_ptr<SolverParam> adadelta_solver(float learning_rate = 10E-2f, float momentum = 0.9f, float delta = 1e-8f, std::string name = "adadelta");
	
	// PRINTERS
	void print(std::initializer_list<std::string> inputs, std::string message, Print::PrintTime = Print::PrintTime::END_OF_EPOCH, Print::PrintType = Print::VALUES, std::string name = "print", std::initializer_list<std::string> phases = {});

	// UTILITIES
	void save_as_binary(std::string filePath);
	void load_from_binary(std::string filePath);
	void save_as_text(std::string filePath);	
	void define_phase(std::string phase, PhaseParam_PhaseBehaviour behaviour);	

	std::shared_ptr<GraphParam> graph();
	std::shared_ptr<Session> session();

private:
	std::string _get_unique_node_name(const std::string &prefix) const;
	std::string _get_unique_solver_name(const std::string &prefix) const;
	std::shared_ptr<NodeParam> _find_node_by_name(const std::string &name) const;
	std::string _reduce(std::string input, int reduce_dimention, ReduceParam_ReduceOp op, int output, std::string name, std::initializer_list<std::string> phases);
private:
	std::list<std::shared_ptr<NodeParam>> _nodes;	
	std::list<std::shared_ptr<SolverParam>> _solvers;	
	std::list<std::shared_ptr<PhaseParam>> _phases;
	
};