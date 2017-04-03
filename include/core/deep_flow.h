#pragma once

#include "core/export.h"
#include "core/variable.h"
#include "core/block.h"

#include "readers/mnist_reader.h"

#include "ops/add.h"
#include "ops/matmul.h"
#include "ops/relu.h"
#include "ops/softmax.h"
#include "ops/square.h"
#include "ops/bias_add.h"
#include "ops/softmax_loss.h"
#include "ops/dropout.h"
#include "ops/convolution_2d.h"
#include "ops/pooling.h"
#include "ops/reduce.h"
#include "ops/equal.h"

#include "initializers/fill.h"
#include "initializers/index_fill.h"
#include "initializers/random_uniform.h"
#include "initializers/step.h"

#include "solvers/sgd_solver.h"
#include "solvers/gain_solver.h"

class DeepFlowDllExport DeepFlow {
public:
	// READERS
	std::shared_ptr<MNISTReader> mnist_reader(std::string folder_path, int batch_size, MNISTReaderType type, std::string name = "MNIST");

	// INITIALIZERS
	std::shared_ptr<Fill> fill(std::initializer_list<int> dims, float value, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<IndexFill> index_fill(std::initializer_list<int> dims, float offset, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Fill> zeros(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Fill> ones(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<RandomUniform> random_uniform(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Step> step(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);

	// VARIABLES
	std::shared_ptr<OutputTerminal> variable(std::shared_ptr<Initializer> initializer, std::string name = "Variable");
	std::shared_ptr<OutputTerminal> variable(std::shared_ptr<Initializer> initializer, int interval , std::string prefix, int perImageHeight, int perImageWidth ,  std::string name = "VariableWithSnapshot");

	// OPS
	std::shared_ptr<OutputTerminal> add(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "Add");
	std::shared_ptr<OutputTerminal> bias_add(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "Bias");
	std::shared_ptr<OutputTerminal> subtract(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "Subtract");
	std::shared_ptr<OutputTerminal> softmax(std::shared_ptr<OutputTerminal> a, std::string name = "Softmax");
	std::shared_ptr<OutputTerminal> square(std::shared_ptr<OutputTerminal> a, std::string name = "Square");
	std::shared_ptr<OutputTerminal> matmul(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "InnerProduct");
	std::shared_ptr<OutputTerminal> relu(std::shared_ptr<OutputTerminal> a, float negative_slope = -0.01f, std::string name = "Relu");
	std::shared_ptr<OutputTerminal> dropout(std::shared_ptr<OutputTerminal> a, float dropout = 0.5f, std::string name = "Dropout");
	std::shared_ptr<OutputTerminal> conv2d(std::shared_ptr<OutputTerminal> input, std::shared_ptr<OutputTerminal> filter, std::string name = "Conv");
	std::shared_ptr<OutputTerminal> pooling(std::shared_ptr<OutputTerminal> input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "MaxPool");
	std::shared_ptr<OutputTerminal> argmax(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name = "Argmax");
	std::shared_ptr<OutputTerminal> reduce_max(std::shared_ptr<OutputTerminal> input, int reduceDimension, std::string name = "Max");
	std::shared_ptr<OutputTerminal> equal(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "Equal");

	// BLOCK
	std::shared_ptr<Block> block(std::initializer_list<std::shared_ptr<InputTerminal>> inputs, std::initializer_list<std::shared_ptr<OutputTerminal>> outputs, std::string name = "Block");	

	// LOSS
	std::shared_ptr<OutputTerminal> softmax_loss(std::shared_ptr<OutputTerminal> a, std::shared_ptr<OutputTerminal> b, std::string name = "SoftmaxLoss");
	
	
	// SOLVERS
	std::shared_ptr<SGDSolver> sgd_solver(std::shared_ptr<OutputTerminal> loss, int max_iteration, float momentum, float learning_rate);
	std::shared_ptr<GainSolver> gain_solver(std::shared_ptr<OutputTerminal> loss, int max_iteration = 1000, float momentum = 0.99999, float learning_rate = 0.0001, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f);
};
