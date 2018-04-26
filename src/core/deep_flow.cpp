#include "core/deep_flow.h"

#include "core/session.h"
#include "core/caffe.h"

void add_outputs(deepflow::NodeParam *node_param, int num_outputs) {
	for (int i = 0; i < num_outputs; ++i)
		node_param->add_output(node_param->name() + "_output_" + std::to_string(i));
}

void add_scope(deepflow::NodeParam *node_param, std::string df_scope, std::string node_scope ) {
	node_param->set_scope((!node_scope.empty() && node_scope != "Default") ? node_scope : df_scope);
}

void add_scope(deepflow::SolverParam *solver_param, std::string df_scope, std::string node_scope) {
	solver_param->set_scope((!node_scope.empty() && node_scope != "Default") ? node_scope : df_scope);
}

DeepFlow::DeepFlow()
{		
	_block = std::make_shared<Block>();
}

DeepFlow::DeepFlow(std::shared_ptr<Block> block)
{
	_block = block;
}

std::string DeepFlow::mnist_reader(std::string folder_path, MNISTReaderOp &params) {	
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto mnistParam = node_param->mutable_mnist_param();
	mnistParam->set_folder_path(folder_path);
	mnistParam->set_reader_type((deepflow::MnistParam::ReaderType) params._dataset);
	mnistParam->set_output_type((deepflow::MnistParam::OutputType) params._output);
	mnistParam->set_batch_size(params._batch_size);
	return node_param->output(0);

}

std::string DeepFlow::data_generator(std::string initializer, DataGeneratorOp &params)
{	
	auto node_param = _block->add_node_param();	
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto variable_param = node_param->mutable_variable_param();	
	auto data_generator_param = node_param->mutable_data_generator_param();	
	if (!params._solver.empty())
		variable_param->set_solver_name(params._solver);
	auto init_param = variable_param->mutable_init_param();
	init_param->CopyFrom(*_block->find_initializer_param_by_name(initializer));	
	return node_param->output(0);
}

std::array<std::string, 2> DeepFlow::text_image_generator(std::string initializer, TextImageGeneratorOp &params)
{	
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 2);
	auto variable_param = node_param->mutable_variable_param();
	auto text_image_generator_param = node_param->mutable_text_image_generator_param();
	text_image_generator_param->set_chars(params._chars);	
	for (auto word : params._words) {
		text_image_generator_param->add_words(word);		
	}
	auto init_param = variable_param->mutable_init_param();
	init_param->CopyFrom(*_block->find_initializer_param_by_name(initializer));
	std::array<std::string, 2> outputs;
	outputs[0] = node_param->output(0);
	outputs[1] = node_param->output(1);
	return outputs;
}

std::string DeepFlow::imread(std::string file_path, ImreadOp &params)
{	
	auto node_param = _block->add_node_param();	
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto image_reader_param = node_param->mutable_image_reader_param();
	image_reader_param->set_file_name(file_path);
	image_reader_param->set_type((deepflow::ImageReaderParam_Type) params._type);	
	return node_param->output(0);
}

std::string DeepFlow::imbatch(std::string folder_path, std::initializer_list<int> dims, ImbatchOp &params)
{	
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto image_batch_reader_param = node_param->mutable_image_batch_reader_param();	
	image_batch_reader_param->set_randomize(params._randomize);
	image_batch_reader_param->set_folder_path(folder_path);	
	image_batch_reader_param->set_between_0_and_1(params._between_0_and_1);
	std::vector<int> values(dims);
	auto tensor_param = image_batch_reader_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);	
	return node_param->output(0);
}

std::string DeepFlow::add(std::string a, std::string b, AddOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	auto add_param = node_param->mutable_add_param();
	add_param->set_alpha(params._alpha);
	add_param->set_beta(params._beta);
	return node_param->output(0);
}

std::string DeepFlow::dot(std::string a, std::string b, DotOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	node_param->mutable_dot_param();
	return node_param->output(0);
}

std::string DeepFlow::subtract(std::string a, std::string b, SubtractOp &params) {
	return add(a, b, AddOp().name(params._name).scope(params._scope).alpha(1.0f).beta(-1.0f));
}

std::string DeepFlow::bias_add(std::string input, std::string weights, BiasAddOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(weights);
	auto bias_add_param = node_param->mutable_bias_add_param();	
	return node_param->output(0);
}

std::string DeepFlow::bias_add(std::string input, int output_channels, std::string solver, BiasAddOp &params)
{
	auto bnb = variable(zeros({ 1, output_channels, 1, 1 }), solver, VariableOp(params._name + "_bias_w").scope(params._scope));
	return bias_add(input, bnb, params);
}

std::string DeepFlow::index_fill(std::initializer_list<int> dims, float offset, std::string name) {
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto index_fill_param = init_param->mutable_index_fill_param();
	index_fill_param->set_offset(offset);
	return init_param->name();
}

std::string DeepFlow::step(std::initializer_list<int> dims, float min, float max, std::string name) {
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto step_param = init_param->mutable_step_param();
	step_param->set_max(max);
	step_param->set_min(min);
	return init_param->name();
}

std::string DeepFlow::three_state(std::initializer_list<int> dims, std::string name)
{
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto three_state_param = init_param->mutable_three_state_param();
	return init_param->name();
}

std::string DeepFlow::gradient(std::initializer_list<int> dims, std::string name)
{
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	init_param->mutable_gradient_fill_param();
	return init_param->name();
}

std::string DeepFlow::constant(std::initializer_list<int> dims, std::vector<float> values, std::string name)
{
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));	
	auto tensor_param = init_param->mutable_tensor_param();
	for (auto dim : dims)
		tensor_param->add_dims(dim);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto constant_param = init_param->mutable_constant_param();
	for (auto value : values)
		constant_param->add_values(value);
	return init_param->name();
}

std::string DeepFlow::fill(std::initializer_list<int> dims, float value, std::string name) {
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto fill_param = init_param->mutable_fill_param();
	fill_param->set_value(value);
	return init_param->name();
}

std::string DeepFlow::zeros(std::initializer_list<int> dims, std::string name) {
	return fill(dims, 0.0f, name);
}

std::string DeepFlow::ones(std::initializer_list<int> dims, std::string name) {
	return fill(dims, 1.0f, name);
}

std::string DeepFlow::random_uniform(std::initializer_list<int> dims,  float min, float max, std::string name) {
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);	
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto random_uniform_param = init_param->mutable_random_uniform_param();
	random_uniform_param->set_max(max);
	random_uniform_param->set_min(min);
	return init_param->name();
}

std::string DeepFlow::random_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name)
{
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);	
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto random_normal_param = init_param->mutable_random_normal_param();
	random_normal_param->set_mean(mean);
	random_normal_param->set_stddev(stddev);
	return init_param->name();
}

std::string DeepFlow::truncated_normal(std::initializer_list<int> dims, float mean, float stddev, std::string name)
{
	auto init_param = _block->add_initializer_param();
	init_param->set_name(_block->get_unique_initializer_param_name(name));
	std::vector<int> values(dims);
	auto tensor_param = init_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	auto t_normal_param = init_param->mutable_truncated_normal_param();
	t_normal_param->set_mean(mean);
	t_normal_param->set_stddev(stddev);
	return init_param->name();
}

std::string DeepFlow::cast_float(std::string input, CastFloatOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);	
	node_param->mutable_cast_float_param();
	return node_param->output(0);
}

std::string DeepFlow::softmax(std::string a, SoftmaxOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto softmax_param = node_param->mutable_softmax_param();
	softmax_param->set_mode((deepflow::SoftmaxParam_Mode) params._mode);
	return node_param->output(0);
}

std::string DeepFlow::max(std::string a, std::string b, MaxOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	node_param->mutable_max_param();	
	return node_param->output(0);
}

std::string DeepFlow::nand(std::string a, std::string b, NandOp & params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	node_param->mutable_nand_param();
	return node_param->output(0);
}

std::string DeepFlow::square(std::string a, SquareOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto square_param = node_param->mutable_square_param();
	return node_param->output(0);
}

std::string DeepFlow::abs(std::string a, AbsOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto abs_param = node_param->mutable_abs_param();
	return node_param->output(0);
}

std::string DeepFlow::exp(std::string a, ExpOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto square_param = node_param->mutable_exp_param();
	return node_param->output(0);
}

std::string DeepFlow::log(std::string a, LogOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto log_param = node_param->mutable_log_param();
	log_param->set_coef(params._coef);
	return node_param->output(0);
}

std::string DeepFlow::place_holder(std::array<int, 4> dims, PlaceholderOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto place_holder_param = node_param->mutable_place_holder_param();
	auto tensor_param = place_holder_param->mutable_tensor_param();
	tensor_param->set_type((deepflow::TensorParam_TensorType)params._type);
	for (int i = 0; i < 4; ++i)
		tensor_param->add_dims(dims[i]);	
	return node_param->output(0);
}

std::string DeepFlow::restructure(std::string input, int first_dim, int second_dim, RestructureOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto restructure_param = node_param->mutable_restructure_param();
	restructure_param->set_first_dim(first_dim);
	restructure_param->set_second_dim(second_dim);
	return node_param->output(0);
}

std::string DeepFlow::variable(std::string initializer, std::string solver, VariableOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto variable_param = node_param->mutable_variable_param();
	if (!solver.empty())
		variable_param->set_solver_name(solver);
	auto init_param = variable_param->mutable_init_param();
	auto init = _block->find_initializer_param_by_name(initializer);
	init_param->CopyFrom(*init);
	return node_param->output(0);
}

std::string DeepFlow::matmul(std::string a, std::string b, MatmulOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	auto matmul_param = node_param->mutable_matmul_param();
	return node_param->output(0);
}

std::string DeepFlow::dense(std::string input, std::initializer_list<int> dims, std::string solver, DenseOp &params)
{
	auto w = variable(truncated_normal(dims, 0, params._stddev), solver, VariableOp(params._name + "_fc_w").scope(params._scope));
	std::string node;
	if (params._with_bias) {
		node = matmul(input, w, MatmulOp(params._name + "_fc").scope(params._scope));
		node = bias_add(node, (*(dims.begin() + 1)), solver, BiasAddOp(params._name).scope(params._scope));
	}
	else {
		node = matmul(input, w, MatmulOp(params._name).scope(params._scope));		
	}
	return node;	
}

std::string DeepFlow::leaky_relu(std::string a, LeakyReluOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);	
	auto relu_param = node_param->mutable_leaky_relu_param();
	relu_param->set_negative_slope(params._negative_slope);
	return node_param->output(0);
}

std::string DeepFlow::sigmoid(std::string a, SigmoidOp &params)
{
	auto node_param = _block->add_node_param();	
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_SIGMOID);	
	return node_param->output(0);	
}

std::string DeepFlow::relu(std::string a, ReluOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_RELU);	
	return node_param->output(0);	
}

std::string DeepFlow::prelu(std::string input, std::string w, PReluOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(w);
	auto prelu_param = node_param->mutable_prelu_param();
	return node_param->output(0);
}

std::string DeepFlow::prelu(std::string input, int output_channels, std::string solver, PReluOp &params)
{
	auto w = variable(fill({ 1, output_channels, 1, 1 }, params._initial_slope), solver, VariableOp(params._name + "_w").scope(params._scope));
	return prelu(input, w, PReluOp(params._name));	
}

std::string DeepFlow::dprelu(std::string input, std::string enabler, DPReluOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(enabler);
	auto dprelu_param = node_param->mutable_dprelu_param();
	dprelu_param->set_negative_slope(params._negative_slope);
	return node_param->output(0);
}

std::string DeepFlow::tanh(std::string a, TanhOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_TANH);	
	return node_param->output(0);	
}

std::string DeepFlow::clipped_relu(std::string a, ClippedReluOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(params._threshold);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_CLIPPED_RELU);	
	return node_param->output(0);	
}

std::string DeepFlow::elu(std::string a, EluOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(params._alpha);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_ELU);	
	return node_param->output(0);	
}

std::array<std::string, 2> DeepFlow::accumulator(std::string input, AccumulatorOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 2);
	node_param->add_input(input);	
	auto accumulator_param = node_param->mutable_accumulator_param();
	std::array<std::string, 2> outputs;
	outputs[0] = node_param->output(0);
	outputs[1] = node_param->output(1);
	return outputs;	
}

std::string DeepFlow::replay_memory(std::string input, int capacity, ReplayMemoryOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto replay_memory_param = node_param->mutable_replay_memory_param();
	replay_memory_param->set_capacity(capacity);
	return node_param->output(0);
}

std::string DeepFlow::batch_stddev(std::string input, BatchStddevOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->mutable_batch_stddev_param();
	return node_param->output(0);
}

std::string DeepFlow::pass_through(std::string input, PassThroughOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto pass = node_param->mutable_pass_through_param();
	pass->set_stop_gradients(params._stop_gradients);
	return node_param->output(0);
}

std::string DeepFlow::gaussian(std::string mean, std::string sigma, GaussianOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(mean);
	node_param->add_input(sigma);
	node_param->mutable_gaussian_param();
	return node_param->output(0);
}

std::string DeepFlow::gaussian_kernel(int window_size, float sigma, GaussianKernelOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	auto param = node_param->mutable_gaussian_kernel_param();
	param->set_sigma(sigma);
	param->set_window_size(window_size);
	return node_param->output(0);
}

std::string DeepFlow::gaussian_blur(std::string input, int window_size, float sigma, GaussianBlurOp &params)
{
	LOG_IF(FATAL, window_size % 2 == 0) << "Window size for Gaussian Blur kernel must be odd.";
	auto pad = floor(window_size / 2);
	auto k = gaussian_kernel(window_size, sigma, GaussianKernelOp(params._name + "_kernel").scope(params._scope));
	return conv2d(input, k, ConvolutionOp(params._name).pad(pad).scope(params._scope));
}

std::string DeepFlow::identify(std::string input, int input_channels, int output_channels, float scale, std::string scope)
{
	std::vector<float> weights(input_channels * output_channels);
	for (int i = 0; i < input_channels * output_channels; ++i)
		weights[i] = (i % output_channels) % input_channels ? 1.0f : 0.0f;
	auto f = variable(constant({ output_channels, input_channels, 1, 1 }, weights), "", VariableOp().scope(scope));
	std::string node = input;
	if (scale < 1.0)
		node = resize(node, scale, scale, ResizeOp().scope(scope));
	node = conv2d(node, f, ConvolutionOp().scope(scope).kernel(1).pad(0));
	if (scale > 1.0)
		node = resize(node, scale, scale, ResizeOp().scope(scope));
	return node;
}

void DeepFlow::print(std::list<std::string> inputs, std::string message, PrintOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);	
	auto print_param = node_param->mutable_print_param();	
	print_param->set_message(message);
	print_param->set_num_inputs(inputs.size());
	int type;
	if (params._values && params._values)
		type = 2;
	else if (params._values)
		type = 0;
	else
		type = 1;
	print_param->set_print_type((deepflow::ActionType)type);
}

void DeepFlow::logger(std::list<std::string> inputs, std::string file_path, std::string message, LoggerOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);
	auto logger_param = node_param->mutable_logger_param();
	logger_param->set_message(message);
	logger_param->set_num_inputs(inputs.size());
	logger_param->set_file_path(file_path);
	int type;
	if (params._values && params._values)
		type = 2;
	else if (params._values)
		type = 0;
	else
		type = 1;
	logger_param->set_logging_type((deepflow::ActionType)type);
}

std::string DeepFlow::display(std::string input, DisplayOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto display_param = node_param->mutable_display_param();
	display_param->set_delay_msec(params._delay_msec);
	int type;
	if (params._values)
		type = 0;
	else
		type = 1;
	display_param->set_display_type((deepflow::ActionType)type);
	display_param->set_draw_iteration(params._draw_iterations);
	display_param->set_epoch_frequency(params._epoch_freuqency);
	display_param->set_iter_frequency(params._iter_frequency);
	return node_param->output(0);
}

std::string DeepFlow::imwrite(std::string input, std::string filename, ImwriteOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto writer_param = node_param->mutable_image_writer_param();
	writer_param->set_filename(filename);	
	return node_param->output(0);
}

void DeepFlow::psnr(std::string a, std::string b, PsnrOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	node_param->add_input(a);
	node_param->add_input(b);
	auto psnr_param = node_param->mutable_psnr_param();	
}

std::string DeepFlow::batch_normalization(std::string input, std::string scale, std::string bias, BatchNormalizationOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(scale);
	node_param->add_input(bias);
	auto batch_norm_param = node_param->mutable_batch_normalization_param();
	batch_norm_param->set_mode((deepflow::BatchNormalizationParam_Mode) params._mode);
	batch_norm_param->set_cache_meanvar(params._cache);
	batch_norm_param->set_exp_avg_factor(params._exponent_average_factor);
	batch_norm_param->set_eps(params._eps);
	return node_param->output(0);
}

std::string DeepFlow::batch_normalization(std::string input, int output_channels, std::string solver, BatchNormalizationOp &params)
{
	auto bns = variable(fill({ 1, output_channels, 1, 1 }, 1), solver, VariableOp(params._name + "_s").scope(params._scope));
	auto bnb = variable(fill({ 1, output_channels, 1, 1 }, 0), solver, VariableOp(params._name + "_b").scope(params._scope));
	return batch_normalization(input, bns, bnb, BatchNormalizationOp(params._name).scope(params._scope).spatial());
}

std::string DeepFlow::batch_normalization(std::string input, int output_channels, int output_height, int output_width, std::string solver, BatchNormalizationOp & params)
{
	auto bns = variable(fill({ 1, output_channels, output_height, output_width }, 1), solver, VariableOp(params._name + "_s").scope(params._scope));
	auto bnb = variable(fill({ 1, output_channels, output_height, output_width }, 0), solver, VariableOp(params._name + "_b").scope(params._scope));
	return batch_normalization(input, bns, bnb, BatchNormalizationOp(params._name).scope(params._scope).per_activation());
}

std::string DeepFlow::lrn(std::string input, LrnOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto lrn_param = node_param->mutable_lrn_param();
	lrn_param->set_alpha(params._alpha);
	lrn_param->set_beta(params._beta);
	lrn_param->set_k(params._k);
	lrn_param->set_n(params._n);
	return node_param->output(0);
}

void DeepFlow::sio_output(std::list<std::string> inputs, SioOutputOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);
	auto sio_output_param = node_param->mutable_sio_output_param();	
	sio_output_param->set_num_inputs(inputs.size());	
	sio_output_param->set_host(params._host);
	sio_output_param->set_port(params._port);
}


std::string DeepFlow::square_error(std::string a, std::string b, SquareErrorOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);	
	node_param->mutable_square_error_param();
	return node_param->output(0);
}

std::string DeepFlow::sgd_solver( SgdSolverOp &params) {
	auto solver_param = _block->add_solver_param();
	add_scope(solver_param, _scope, params._scope);
	solver_param->set_learning_rate(params._learning_rate);
	solver_param->set_name(_block->get_unique_solver_param_name(params._name));
	auto sgd_solver = solver_param->mutable_sgd_solver();	
	sgd_solver->set_momentum(params._momentum);
	return solver_param->name();
}

std::string DeepFlow::adam_solver( AdamSolverOp &params ) {
	auto solver_param = _block->add_solver_param();
	add_scope(solver_param, _scope, params._scope);
	solver_param->set_learning_rate(params._learning_rate);	
	solver_param->set_name(_block->get_unique_solver_param_name(params._name));
	auto adam_solver = solver_param->mutable_adam_solver();		
	adam_solver->set_beta1(params._beta1);
	adam_solver->set_beta2(params._beta2);
	adam_solver->set_eps(params._eps);
	return solver_param->name();
}

std::string DeepFlow::adadelta_solver( AdaDeltaSolverOp &params ) {
	auto solver_param = _block->add_solver_param();
	add_scope(solver_param, _scope, params._scope);
	solver_param->set_learning_rate(params._learning_rate);
	solver_param->set_name(_block->get_unique_solver_param_name(params._name));
	auto adadelta_solver = solver_param->mutable_adadelta_solver();	
	adadelta_solver->set_momentum(params._momentum);
	adadelta_solver->set_delta(params._delta);	
	return solver_param->name();
}

std::string DeepFlow::rmsprop_solver( RmsPropSolverOp &params)
{
	auto solver_param = _block->add_solver_param();
	add_scope(solver_param, _scope, params._scope);
	solver_param->set_learning_rate( params._learning_rate);
	solver_param->set_name(_block->get_unique_solver_param_name(params._name));
	auto solver = solver_param->mutable_rmsprop_solver();
	solver->set_rms_decay(params._rms_decay);
	solver->set_eps(params._eps);
	return solver_param->name();
}

std::string DeepFlow::dropout(std::string a, DropoutOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	auto dropout_param = node_param->mutable_dropout_param();	
	dropout_param->set_dropout(params._ratio);	
	return node_param->output(0);
}

std::string DeepFlow::pooling(std::string input, PoolingOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto pooling_param = node_param->mutable_pooling_param();
	pooling_param->set_h_pad(params._h_pad);
	pooling_param->set_v_pad(params._v_pad);
	pooling_param->set_h_stride(params._h_stride);
	pooling_param->set_v_stride(params._v_stride);
	pooling_param->set_window_h(params._window_h);
	pooling_param->set_window_w(params._window_w);	
	return node_param->output(0);
}

std::string DeepFlow::transposed_conv2d(std::string input, std::string filter, ConvolutionOp & params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(filter);
	auto tconv_2d_param = node_param->mutable_transposed_conv_2d_param();
	tconv_2d_param->set_pad_h(params._pad_h);
	tconv_2d_param->set_pad_w(params._pad_w);
	tconv_2d_param->set_u(params._stride_u);
	tconv_2d_param->set_v(params._stride_v);
	tconv_2d_param->set_dilation_h(params._dilation_h);
	tconv_2d_param->set_dilation_w(params._dilation_w);	
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, std::string filter, ConvolutionOp & params)
{	
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(filter);
	auto conv_2d_param = node_param->mutable_conv_2d_param();
	conv_2d_param->set_pad_h(params._pad_h);
	conv_2d_param->set_pad_w(params._pad_w);
	conv_2d_param->set_u(params._stride_u);
	conv_2d_param->set_v(params._stride_v);
	conv_2d_param->set_dilation_h(params._dilation_h);
	conv_2d_param->set_dilation_w(params._dilation_w);	
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, int input_channels, int output_channels, std::string solver, ConvolutionOp & params)
{
	std::string name = params._name;
	auto f = variable(truncated_normal({ output_channels, input_channels, params._kernel_h, params._kernel_w }, 0, params._stddev), solver, VariableOp(name + "_conv_f").scope(params._scope));
	std::string node; 
	if (params._with_bias) {
		node = conv2d(input, f, params.name(name + "_conv"));
		node = bias_add(node, output_channels, solver, BiasAddOp(name).scope(params._scope));
	}
	else {
		node = conv2d(input, f, params.name(name));
	}
	return node;
}

std::string DeepFlow::transposed_conv2d(std::string input, int input_channels, int output_channels, std::string solver, ConvolutionOp & params)
{
	std::string name = params._name;
	std::string initializer;
	if (params._stddev == 0)
		initializer = fill({ input_channels, output_channels, params._kernel_h, params._kernel_w }, 0);
	else
		initializer = truncated_normal({ input_channels, output_channels, params._kernel_h, params._kernel_w }, 0, params._stddev);
	auto f = variable(initializer, solver, VariableOp(name + "_tconv_f").scope(params._scope));
	std::string node; 
	if (params._with_bias) {
		node = transposed_conv2d(input, f, params.name(name + "_tconv"));
		node = bias_add(node, output_channels, solver, BiasAddOp(name).scope(params._scope));
	}
	else {
		node = transposed_conv2d(input, f, params.name(name));
	}
	return node;
}

std::string DeepFlow::lifting(std::string input, LiftingOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto lifting_param = node_param->mutable_lifting_param();
	int mode = 0;
	if (params._with_flip) {
		if (params._up)
			mode = 2;
		else
			mode = 3;
	}
	else {
		if (params._up)
			mode = 0;
		else
			mode = 1;
	}
	lifting_param->set_mode((deepflow::LiftingParam_Mode) mode);
	return node_param->output(0);
}

std::string DeepFlow::patching(std::string input, PatchingOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto patching_param = node_param->mutable_patching_param();
	int mode = 0;
	if (params._samples) {
		if (params._up)
			mode = 0;
		else
			mode = 1;
	}
	else {
		if (params._up)
			mode = 2;
		else
			mode = 3;
	}
	patching_param->set_mode((deepflow::PatchingParam_Mode) mode);
	patching_param->set_num_horizontal_patch(params._h_num);
	patching_param->set_num_vertical_patch(params._v_num);
	return node_param->output(0);
}

std::string DeepFlow::patch_sampling(std::string input, int patch_width, int patch_height, PatchSamplingOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto p = node_param->mutable_patch_sampling_param();
	p->set_patch_height(patch_height);
	p->set_patch_width(patch_width);
	return node_param->output(0);
}

std::string DeepFlow::resize(std::string input, float height_scale, float width_scale, ResizeOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto resize_param = node_param->mutable_resize_param();
	resize_param->set_height_scale(height_scale);
	resize_param->set_width_scale(width_scale);
	return node_param->output(0);
}

std::string DeepFlow::concate(std::list<std::string> inputs, ConcateOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	for (auto input : inputs)
		node_param->add_input(input);	
	auto concate_param = node_param->mutable_concate_param();
	concate_param->set_num_inputs(inputs.size());
	return node_param->output(0);
}

std::string DeepFlow::reshape(std::string input, std::array<int,4> output_dims, ReshapeOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto reshape_param = node_param->mutable_reshape_param();
	for (int i = 0; i < 4; ++i)
		reshape_param->add_output_dims(output_dims[i]);
	return node_param->output(0);
}

std::string DeepFlow::spatial_transformer(std::string input, std::string theta, std::string grid, SpatialTransformerOp & params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	node_param->add_input(theta);
	node_param->add_input(grid);
	node_param->mutable_spatial_transformer_param();
	return node_param->output(0);
}

std::string DeepFlow::spatial_transformer(std::string input, std::string theta, int n, int h, int w, std::string solver, SpatialTransformerOp & params)
{
	auto grid = variable(fill({ n,h,w,2 }, 0), solver, VariableOp(params._name + "_grid").scope(params._scope));
	return spatial_transformer(input, theta, grid, params);	
}

std::string DeepFlow::_reduce(std::string input, ReduceOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 2);
	node_param->add_input(input);
	auto reduce_param = node_param->mutable_reduce_param();
	reduce_param->set_reduce_dim(params._reduce_dimention);
	reduce_param->set_output_type(params._output_type);
	reduce_param->set_reduce_op(params._op);	
	return node_param->output(params._terminal);
}

std::string DeepFlow::reduce_all(std::string input, ReduceAllOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto reduce_param = node_param->mutable_reduce_all_param();
	reduce_param->set_reduce_op((deepflow::ReduceAllParam_ReduceAllOp) params._op);
	return node_param->output(0);
}

std::string DeepFlow::argmax(std::string input, int reduce_dimension, ArgmaxOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).max().indicies().terminal(1));	
}

std::string DeepFlow::argmin(std::string input, int reduce_dimension, ArgminOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).min().indicies().terminal(1));	
}

std::string DeepFlow::reduce_max(std::string input, int reduce_dimension, ReduceMaxOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).max().values().terminal(0));	
}

std::string DeepFlow::reduce_amax(std::string input, int reduce_dimension, ReduceAmaxOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).amax().values().terminal(0));	
}

std::string DeepFlow::reduce_min(std::string input, int reduce_dimension, ReduceMinOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).min().values().terminal(0));	
}

std::string DeepFlow::reduce_norm1(std::string input, int reduce_dimension, ReduceNorm1Op &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).norm1().values().terminal(0));	
}

std::string DeepFlow::reduce_norm2(std::string input, int reduce_dimension, ReduceNorm2Op &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).norm2().values().terminal(0));	
}

std::string DeepFlow::reduce_sum(std::string input, int reduce_dimension, ReduceSumOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).add().values().terminal(0));
}

std::string DeepFlow::reduce_mean(std::string input, int reduce_dimension, ReduceMeanOp &params) {
	return _reduce(input, ReduceOp(params._name).scope(params._scope).reduce(reduce_dimension).avg().values().terminal(0));	
}

std::string DeepFlow::random_selector(std::string input_1, std::string input_2, RandomSelectorOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input_1);
	node_param->add_input(input_2);
	auto random_selector_param = node_param->mutable_random_selector_param();
	random_selector_param->set_probability(params._ratio);	
	return node_param->output(0);
}

std::string DeepFlow::multiplexer(std::list<std::string> inputs, MultiplexerOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);	
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);	
	auto multiplexer_param = node_param->mutable_multiplexer_param();
	multiplexer_param->set_num_inputs(inputs.size());
	return node_param->output(0);
}

std::string DeepFlow::switcher(std::string input, SwitchOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);	
	node_param->mutable_switch_param();
	return node_param->output(0);
}

std::string DeepFlow::loss(std::string input, LossOp &params)
{
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(input);
	auto loss_param = node_param->mutable_loss_param();
	loss_param->set_reduce_op((deepflow::LossParam_ReduceOp) params._type);
	loss_param->set_alpha(params._alpha);
	loss_param->set_beta(params._beta);
	return node_param->output(0);
}

std::string DeepFlow::equal(std::string a, std::string b, EqualOp &params) {
	auto node_param = _block->add_node_param();
	add_scope(node_param, _scope, params._scope);
	node_param->set_name(_block->get_unique_node_param_name(params._name));
	add_outputs(node_param, 1);
	node_param->add_input(a);
	node_param->add_input(b);
	node_param->mutable_equal_param();	
	return node_param->output(0);
}

std::shared_ptr<Block> DeepFlow::block() {
	return _block;
}

std::shared_ptr<Session> DeepFlow::session()
{	
	auto session = std::make_shared<Session>(_block);	
	return session;
}

void DeepFlow::load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose)
{
	Caffe caffe(this, verbose);
	caffe.load(file_path, inputs);
}

void DeepFlow::with(std::string scope)
{
	_scope = scope;
}
