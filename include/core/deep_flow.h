#pragma once

#include "core/export.h"

#include "readers/mnist_reader.h"
#include "initializers/fill.h"
#include "initializers/index_fill.h"
#include "initializers/random_uniform.h"
#include "initializers/step.h"
#include "ops/print.h"
#include "ops/accumulator.h"

#include "solvers/sgd_solver.h"
#include "solvers/gain_solver.h"

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
	NodeOutputPtr variable(std::shared_ptr<Initializer> initializer, std::string name = "var", std::initializer_list<std::string> phases = {});
	NodeOutputPtr variable(std::shared_ptr<Initializer> initializer, int interval , std::string prefix, int perImageHeight, int perImageWidth ,  std::string name = "var", std::initializer_list<std::string> phases = {});
	NodeOutputPtr place_holder(std::array<int,4> dims, Tensor::TensorType type = Tensor::Float, std::string name = "ph", std::initializer_list<std::string> phases = {});

	// OPS
	NodeOutputPtr add(NodeOutputPtr a, NodeOutputPtr b, std::string name = "add", std::initializer_list<std::string> phases = {});
	NodeOutputPtr bias_add(NodeOutputPtr a, NodeOutputPtr b, std::string name = "bias", std::initializer_list<std::string> phases = {});
	NodeOutputPtr subtract(NodeOutputPtr a, NodeOutputPtr b, std::string name = "sub", std::initializer_list<std::string> phases = {});
	NodeOutputPtr softmax(NodeOutputPtr a, std::string name = "softmax", std::initializer_list<std::string> phases = {});
	NodeOutputPtr square(NodeOutputPtr a, std::string name = "square", std::initializer_list<std::string> phases = {});
	NodeOutputPtr matmul(NodeOutputPtr a, NodeOutputPtr b, std::string name = "ip", std::initializer_list<std::string> phases = {});
	NodeOutputPtr relu(NodeOutputPtr a, float negative_slope = 0.01f, std::string name = "relu", std::initializer_list<std::string> phases = {});
	NodeOutputPtr dropout(NodeOutputPtr a, float dropout = 0.5f, std::string name = "dropout", std::initializer_list<std::string> phases = {});
	NodeOutputPtr conv2d(NodeOutputPtr input, NodeOutputPtr filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation , std::string name = "conv", std::initializer_list<std::string> phases = {});
	NodeOutputPtr conv2d(NodeOutputPtr input, NodeOutputPtr filter, std::string name = "conv", std::initializer_list<std::string> phases = {});
	NodeOutputPtr pooling(NodeOutputPtr input, int windowHeight = 3, int windowWidth = 3, int verticalPadding = 0, int horizontalPadding = 0, int verticalStride = 1, int horizontalStride = 1, std::string name = "maxpool", std::initializer_list<std::string> phases = {});
	NodeOutputPtr argmax(NodeOutputPtr input, int reduceDimension, std::string name = "argmax", std::initializer_list<std::string> phases = {});
	NodeOutputPtr reduce_max(NodeOutputPtr input, int reduceDimension, std::string name = "max", std::initializer_list<std::string> phases = {});
	NodeOutputPtr equal(NodeOutputPtr a, NodeOutputPtr b, std::string name = "Equal", std::initializer_list<std::string> phases = {});
	NodeOutputPtr reduce_mean(NodeOutputPtr input, int reduceDimension, std::string name = "mean", std::initializer_list<std::string> phases = {});
	NodeOutputPtr reduce_sum(NodeOutputPtr input, int reduceDimension, std::string name = "sum", std::initializer_list<std::string> phases = {});
	NodeOutputPtr cast_float(NodeOutputPtr input, std::string name = "float", std::initializer_list<std::string> phases = {});
	NodeOutputPtr accumulator(NodeOutputPtr input, Accumulator::ResetTime resetTime = Accumulator::ResetTime::EndOfEpoch, std::string name = "acc", std::initializer_list<std::string> phases = {});

	// PHASE
	NodeOutputPtr phaseplexer(NodeOutputPtr input_1, std::string phase_1, NodeOutputPtr input_2, std::string phase_2, std::string name = "plex", std::initializer_list<std::string> phases = {});

	// LOSS
	NodeOutputPtr softmax_loss(NodeOutputPtr a, NodeOutputPtr b, std::string name = "softmaxloss", std::initializer_list<std::string> phases = {});
		
	// SOLVERS
	std::shared_ptr<SGDSolver> sgd_solver(NodeOutputPtr loss, int max_epoch, float momentum, float learning_rate);
	std::shared_ptr<GainSolver> gain_solver(NodeOutputPtr loss, int max_epoch = 1000, float momentum = 0.99999, float learning_rate = 0.0001, float max_gain = 10, float min_gain = 0.1, float gain_plus = 0.05f, float gain_mult = 0.95f);
	
	// PRINTERS
	std::shared_ptr<Print> print(std::initializer_list<NodeOutputPtr> inputs, std::string message, Print::PrintTime = Print::PrintTime::END_OF_EPOCH, Print::PrintType = Print::VALUES, std::string name = "print", std::initializer_list<std::string> phases = {});

	// UTILITIES
	void global_node_initializer();
	void eval(NodeOutputPtr terminal, bool restart = true);
	std::tuple<float, float, int> run(NodeOutputPtr loss, NodeOutputPtr accuracy, std::map<NodeOutputPtr, NodeOutputPtr> feed);
	void run(std::string phase, bool print_iteration, bool print_epoch, int debug_level);		
	void save_as_binary(std::string filePath, bool include_inits);
	void load_from_binary(std::string filePath);
	void save_as_text(std::string filePath, bool include_weights = false, bool include_inits = false);
	void define_phase(std::string phase, PhaseParam_PhaseBehaviour behaviour);	
	void set_solver(std::shared_ptr<Solver> solver);

private:
	std::string getUniqueNodeName(const std::string &prefix) const;
	std::shared_ptr<Node> findNodeByName(const std::string &name) const;
	std::shared_ptr<NodeOutput> findNodeOutputByName(const std::string &name) const;
	std::shared_ptr<NetworkParam> createNetworkParam(bool include_weights, bool include_inits);
	template <class T>
	std::list<std::shared_ptr<T>> getNodes(std::string execution_phase);	
	std::list<std::shared_ptr<Node>> getEndNodes(std::string execution_phase);
private:
	std::list<std::shared_ptr<Node>> _nodes;
	std::list<std::shared_ptr<Variable>> _variables;
	std::list<std::shared_ptr<Solver>> _solvers;
	std::shared_ptr<Solver> _solver;
	std::map<std::string, PhaseParam_PhaseBehaviour> _phases;
};

template<class T>
inline std::list<std::shared_ptr<T>> DeepFlow::getNodes(std::string execution_phase)
{
	std::list<std::shared_ptr<T>> list;
	for (auto node : _nodes) {
		if (node->includePhase(execution_phase)) {
			auto node_after_cast = std::dynamic_pointer_cast<T>(node);
			if (node_after_cast)
				list.push_back(node_after_cast);
		}
	}
	return list;
}
