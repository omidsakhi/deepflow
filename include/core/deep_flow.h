#pragma once

#include "core/export.h"
#include "core/variable.h"
#include "core/place_holder.h"
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

#include <unordered_map>
#include <map>

#include<memory>

class DeepFlowDllExport DeepFlow {
	friend class ::Node;
public:
	// READERS
	std::shared_ptr<MNISTReader> mnist_reader(std::string folder_path, int batch_size, MNISTReaderType type, std::string name = "mnist", std::initializer_list<std::string> phases = {});

	// INITIALIZERS
	std::shared_ptr<Fill> fill(std::initializer_list<int> dims, float value, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<IndexFill> index_fill(std::initializer_list<int> dims, float offset, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Fill> zeros(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Fill> ones(std::initializer_list<int> dims, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<RandomUniform> random_uniform(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);
	std::shared_ptr<Step> step(std::initializer_list<int> dims, float min, float max, Tensor::TensorType type = Tensor::Float);

	// VARIABLES & PLACE HOLDERS
	std::shared_ptr<NodeOutput> variable(std::shared_ptr<Initializer> initializer, std::string name = "Variable", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> variable(std::shared_ptr<Initializer> initializer, int interval , std::string prefix, int perImageHeight, int perImageWidth ,  std::string name = "VariableWithSnapshot", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "PlaceHolder", std::initializer_list<std::string> phases = {});

	// OPS
	std::shared_ptr<NodeOutput> add(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "Add", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> bias_add(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "BiasAdd", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> subtract(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "Subtract", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> softmax(std::shared_ptr<NodeOutput> a, std::string name = "Softmax", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> square(std::shared_ptr<NodeOutput> a, std::string name = "Square", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> matmul(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "InnerProduct", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> relu(std::shared_ptr<NodeOutput> a, float negative_slope = -0.01f, std::string name = "Relu", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> dropout(std::shared_ptr<NodeOutput> a, float dropout = 0.5f, std::string name = "Dropout", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> conv2d(std::shared_ptr<NodeOutput> input, std::shared_ptr<NodeOutput> filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation , std::string name = "Conv", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> conv2d(std::shared_ptr<NodeOutput> input, std::shared_ptr<NodeOutput> filter, std::string name = "Conv", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> pooling(std::shared_ptr<NodeOutput> input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "MaxPool", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> argmax(std::shared_ptr<NodeOutput> input, int reduceDimension, std::string name = "ReduceArgmax", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> reduce_max(std::shared_ptr<NodeOutput> input, int reduceDimension, std::string name = "ReduceMax", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> equal(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "Equal", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> reduce_mean(std::shared_ptr<NodeOutput> input, int reduceDimension, std::string name = "ReduceMean", std::initializer_list<std::string> phases = {});
	std::shared_ptr<NodeOutput> reduce_sum(std::shared_ptr<NodeOutput> input, int reduceDimension, std::string name = "ReduceSum", std::initializer_list<std::string> phases = {});

	// BLOCK
	std::shared_ptr<Block> block(std::initializer_list<std::shared_ptr<NodeInput>> inputs, std::initializer_list<std::shared_ptr<NodeOutput>> outputs, std::string name = "Block", std::initializer_list<std::string> phases = {});

	// LOSS
	std::shared_ptr<NodeOutput> softmax_loss(std::shared_ptr<NodeOutput> a, std::shared_ptr<NodeOutput> b, std::string name = "SoftmaxWithLoss", std::initializer_list<std::string> phases = {});
		
	// SOLVERS
	std::shared_ptr<SGDSolver> sgd_solver(std::shared_ptr<NodeOutput> loss, int max_iteration, float momentum, float learning_rate);
	std::shared_ptr<GainSolver> gain_solver(std::shared_ptr<NodeOutput> loss, int max_iteration = 1000, float momentum = 0.99999, float learning_rate = 0.0001, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f);
	
	// UTILITIES
	void global_node_initializer();
	void eval(std::shared_ptr<NodeOutput> terminal, bool restart = true);
	std::tuple<float, float, int> run(std::shared_ptr<NodeOutput> loss, std::shared_ptr<NodeOutput> accuracy, std::map<std::shared_ptr<NodeOutput>, std::shared_ptr<NodeOutput>> feed);
	std::shared_ptr<Node> findNodeByName(const std::string &name) const;
	std::string getUniqueNodeName(const std::string &prefix) const;
	void save_as_binary(std::string filePath);
	void save_as_text(std::string filePath, bool include_weights = false);
	void define_phase(std::string phase);
	std::shared_ptr<NetworkParam> createNetworkParam(bool include_weights);
private:
	std::list<std::shared_ptr<Node>> _nodes;
	std::list<std::shared_ptr<Variable>> _variables;
	std::list<std::shared_ptr<Solver>> _solvers;
	std::set<std::string> _phases;
};

