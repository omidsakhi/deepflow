#include "core/deep_flow.h"

#include "core/session.h"
#include "core/caffe.h"

void add_outputs(deepflow::NodeParam *node_param, int num_outputs) {
	for (int i = 0; i < num_outputs; ++i)
		node_param->add_output(node_param->name() + "_output_" + std::to_string(i));
}

DeepFlow::DeepFlow()
{		
	_block = std::make_shared<Block>();
}

DeepFlow::DeepFlow(std::shared_ptr<Block> block)
{
	_block = block;
}

std::string DeepFlow::mnist_reader(std::string folder_path, int batch_size, MNISTReader::MNISTReaderType reader_type, MNISTReader::MNISTOutputType output_type, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);	
	auto mnistParam = node_param->mutable_mnist_param();
	mnistParam->set_folder_path(folder_path);
	mnistParam->set_reader_type((deepflow::MnistParam::ReaderType) reader_type);
	mnistParam->set_output_type((deepflow::MnistParam::OutputType) output_type);
	mnistParam->set_batch_size(batch_size);
	return node_param->output(0);

}

std::string DeepFlow::data_generator(std::string initializer, int num_samples, std::string solver, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto variable_param = node_param->mutable_variable_param();	
	auto data_generator_param = node_param->mutable_data_generator_param();	
	data_generator_param->set_num_samples(num_samples);
	if (!solver.empty())
		variable_param->set_solver_name(solver);
	auto init_param = variable_param->mutable_init_param();
	init_param->CopyFrom(*_block->find_initializer_param_by_name(initializer));	
	return node_param->output(0);
}

std::string DeepFlow::image_reader(std::string file_path, deepflow::ImageReaderParam_Type type, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);	
	auto image_reader_param = node_param->mutable_image_reader_param();
	image_reader_param->set_file_name(file_path);
	image_reader_param->set_type(type);	
	return node_param->output(0);
}

std::string DeepFlow::image_batch_reader(std::string folder_path, std::initializer_list<int> dims, bool randomize, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto image_batch_reader_param = node_param->mutable_image_batch_reader_param();	
	image_batch_reader_param->set_randomize(randomize);
	image_batch_reader_param->set_folder_path(folder_path);	

	std::vector<int> values(dims);
	auto tensor_param = image_batch_reader_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);	
	return node_param->output(0);
}

std::string DeepFlow::add(std::string a, std::string b, float alpha, float beta, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto add_param = node_param->mutable_add_param();
	add_param->set_alpha(alpha);
	add_param->set_beta(beta);
	return node_param->output(0);
}

std::string DeepFlow::add(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	return add(a, b, 1.0f, 1.0f, name, phases);
}

std::string DeepFlow::dot(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	node_param->mutable_dot_param();
	return node_param->output(0);
}

std::string DeepFlow::subtract(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	return add(a, b, 1.0f, -1.0f, name, phases);
}

std::string DeepFlow::bias_add(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto bias_add_param = node_param->mutable_bias_add_param();	
	return node_param->output(0);
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

std::string DeepFlow::cast_float(std::string input, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);	
	node_param->mutable_cast_float_param();
	return node_param->output(0);
}

std::string DeepFlow::negate(std::string input, ActionType negateType, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto negate = node_param->mutable_negate_param();
	negate->set_negate_type((deepflow::ActionType)negateType);
	return node_param->output(0);
}

std::string DeepFlow::softmax(std::string a, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto softmax_param = node_param->mutable_softmax_param();
	return node_param->output(0);
}

std::string DeepFlow::square(std::string a, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto square_param = node_param->mutable_square_param();
	return node_param->output(0);
}

std::string DeepFlow::abs(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto abs_param = node_param->mutable_abs_param();
	return node_param->output(0);
}

std::string DeepFlow::exp(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto square_param = node_param->mutable_exp_param();
	return node_param->output(0);
}

std::string DeepFlow::log(std::string a, float coef, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto log_param = node_param->mutable_log_param();
	log_param->set_coef(coef);
	return node_param->output(0);
}

std::string DeepFlow::place_holder(std::array<int, 4> dims, Tensor::TensorType type, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto place_holder_param = node_param->mutable_place_holder_param();
	auto tensor_param = place_holder_param->mutable_tensor_param();
	tensor_param->set_type((deepflow::TensorParam_TensorType)type);
	for (int i = 0; i < 4; ++i)
		tensor_param->add_dims(dims[i]);	
	return node_param->output(0);
}

std::string DeepFlow::restructure(std::string input, int first_dim, int second_dim, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto restructure_param = node_param->mutable_restructure_param();
	restructure_param->set_first_dim(first_dim);
	restructure_param->set_second_dim(second_dim);
	return node_param->output(0);
}

std::string DeepFlow::variable(std::string initializer, std::string solver, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto variable_param = node_param->mutable_variable_param();
	if (!solver.empty())
		variable_param->set_solver_name(solver);
	auto init_param = variable_param->mutable_init_param();
	auto init = _block->find_initializer_param_by_name(initializer);
	init_param->CopyFrom(*init);
	return node_param->output(0);
}

std::string DeepFlow::matmul(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto matmul_param = node_param->mutable_matmul_param();
	return node_param->output(0);
}

std::string DeepFlow::leaky_relu(std::string a, float negative_slope, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);	
	auto relu_param = node_param->mutable_leaky_relu_param();
	relu_param->set_negative_slope(negative_slope);
	relu_param->set_randomize(false);
	return node_param->output(0);
}

std::string DeepFlow::sigmoid(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_SIGMOID);	
	return node_param->output(0);	
}

std::string DeepFlow::relu(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_RELU);	
	return node_param->output(0);	
}

std::string DeepFlow::tanh(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(0);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_TANH);	
	return node_param->output(0);	
}

std::string DeepFlow::clipped_relu(std::string a, float threshold, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(threshold);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_CLIPPED_RELU);	
	return node_param->output(0);	
}

std::string DeepFlow::elu(std::string a, float alpha, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(alpha);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_ELU);	
	return node_param->output(0);	
}

std::array<std::string, 2> DeepFlow::accumulator(std::string input, ActionTime resetTime, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 2);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);	
	auto accumulator_param = node_param->mutable_accumulator_param();
	accumulator_param->set_reset_time((deepflow::ActionTime)resetTime);
	std::array<std::string, 2> outputs;
	outputs[0] = node_param->output(0);
	outputs[1] = node_param->output(1);
	return outputs;	
}

std::string DeepFlow::replay_memory(std::string input, int capacity, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto replay_memory_param = node_param->mutable_replay_memory_param();
	replay_memory_param->set_capacity(capacity);
	return node_param->output(0);
}

std::string DeepFlow::agc(std::string input, float sensitivity, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);	
	auto agc_param = node_param->mutable_agc_param();
	agc_param->set_sensitivity(sensitivity);	
	return node_param->output(0);
}

void DeepFlow::print(std::initializer_list<std::string> inputs, std::string message, ActionTime printTime, ActionType printType, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);	
	auto print_param = node_param->mutable_print_param();	
	print_param->set_message(message);
	print_param->set_num_inputs(inputs.size());
	print_param->set_print_time((deepflow::ActionTime)printTime);
	print_param->set_print_type((deepflow::ActionType)printType);
}

void DeepFlow::logger(std::initializer_list<std::string> inputs, std::string file_path, std::string message, ActionTime loggingTime, ActionType loggingType, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);
	auto logger_param = node_param->mutable_logger_param();
	logger_param->set_message(message);
	logger_param->set_num_inputs(inputs.size());
	logger_param->set_file_path(file_path);
	logger_param->set_logging_time((deepflow::ActionTime)loggingTime);
	logger_param->set_logging_type((deepflow::ActionType)loggingType);
}

std::string DeepFlow::display(std::string input, int delay_msec, ActionTime dislayTime, ActionType displayType, int epoch_frequency, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto display_param = node_param->mutable_display_param();
	display_param->set_delay_msec(delay_msec);
	display_param->set_display_type((deepflow::ActionType)displayType);
	display_param->set_display_time((deepflow::ActionTime)dislayTime);
	display_param->set_epoch_frequency(epoch_frequency);
	return node_param->output(0);
}

void DeepFlow::psnr(std::string a, std::string b, ActionTime printTime, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto psnr_param = node_param->mutable_psnr_param();
	psnr_param->set_print_time((deepflow::ActionTime)printTime);
}

std::string DeepFlow::batch_normalization(std::string input, std::string scale, std::string bias, NormalizationMode mode, bool cache, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	node_param->add_input(scale);
	node_param->add_input(bias);
	auto batch_norm_param = node_param->mutable_batch_normalization_param();
	batch_norm_param->set_mode((deepflow::BatchNormalizationParam_Mode) mode);
	batch_norm_param->set_cache_meanvar(cache);
	return node_param->output(0);
}

void DeepFlow::sio_output(std::initializer_list<std::string> inputs, ActionTime printTime, std::string host, int port, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);
	auto sio_output_param = node_param->mutable_sio_output_param();	
	sio_output_param->set_num_inputs(inputs.size());
	sio_output_param->set_print_time((deepflow::ActionTime)printTime);
	sio_output_param->set_host(host);
	sio_output_param->set_port(port);
}


std::string DeepFlow::square_error(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();	
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);	
	node_param->mutable_square_error_param();
	return node_param->output(0);
}

std::string DeepFlow::sgd_solver(float momentum, float learning_rate, std::string name, std::string enable_input) {
	auto solver_param = _block->add_solver_param();	
	solver_param->set_enable_input(enable_input);
	solver_param->set_name(_block->get_unique_solver_param_name(name));
	auto sgd_solver = solver_param->mutable_sgd_solver();
	sgd_solver->set_learning_rate(learning_rate);
	sgd_solver->set_momentum(momentum);
	return solver_param->name();
}

std::string DeepFlow::gain_solver(float momentum, float learning_rate, float max_gain, float min_gain, float gain_plus, float gain_mult, std::string name, std::string enable_input) {
	auto solver_param = _block->add_solver_param();	
	solver_param->set_enable_input(enable_input);
	solver_param->set_name(_block->get_unique_solver_param_name(name));
	auto gain_solver = solver_param->mutable_gain_solver();
	gain_solver->set_momentum(momentum);
	gain_solver->set_learning_rate(learning_rate);
	gain_solver->set_max_gain(max_gain);
	gain_solver->set_min_gain(min_gain);
	gain_solver->set_gain_plus(gain_plus);
	gain_solver->set_gain_mult(gain_mult);
	return solver_param->name();
}
std::string DeepFlow::adam_solver(float learning_rate, float beta1, float beta2, float eps, std::string name, std::string enable_input) {
	auto solver_param = _block->add_solver_param();
	solver_param->set_enable_input(enable_input);
	solver_param->set_name(_block->get_unique_solver_param_name(name));
	auto adam_solver = solver_param->mutable_adam_solver();	
	adam_solver->set_learning_rate(learning_rate);
	adam_solver->set_beta1(beta1);
	adam_solver->set_beta2(beta2);
	adam_solver->set_eps(eps);
	return solver_param->name();
}

std::string DeepFlow::adadelta_solver(float learning_rate, float momentum, float delta, std::string name, std::string enable_input) {
	auto solver_param = _block->add_solver_param();
	solver_param->set_enable_input(enable_input);
	solver_param->set_name(_block->get_unique_solver_param_name(name));
	auto adadelta_solver = solver_param->mutable_adadelta_solver();
	adadelta_solver->set_learning_rate(learning_rate);
	adadelta_solver->set_momentum(momentum);
	adadelta_solver->set_delta(delta);	
	return solver_param->name();
}

std::string DeepFlow::dropout(std::string a, float dropout, bool train_only, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto dropout_param = node_param->mutable_dropout_param();	
	dropout_param->set_dropout(dropout);
	dropout_param->set_train_only(train_only);	
	return node_param->output(0);
}

std::string DeepFlow::transposed_conv2d(std::string input, std::string filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	node_param->add_input(filter);
	auto tconv_2d_param = node_param->mutable_transposed_conv_2d_param();
	tconv_2d_param->set_pad_h(pad_top_bottom);
	tconv_2d_param->set_pad_w(pad_left_right);
	tconv_2d_param->set_u(vertical_filter_stride);
	tconv_2d_param->set_v(horizontal_filter_stride);
	tconv_2d_param->set_dilation_h(filter_height_dilation);
	tconv_2d_param->set_dilation_w(filter_width_dialation);	
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, std::string filter, std::string bias, float negative_slope, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	node_param->add_input(filter);
	if (!bias.empty())
		node_param->add_input(bias);
	auto conv_2d_param = node_param->mutable_conv_2d_param();
	conv_2d_param->set_pad_h(pad_top_bottom);
	conv_2d_param->set_pad_w(pad_left_right);
	conv_2d_param->set_u(vertical_filter_stride);
	conv_2d_param->set_v(horizontal_filter_stride);
	conv_2d_param->set_dilation_h(filter_height_dilation);
	conv_2d_param->set_dilation_w(filter_width_dialation);	
	conv_2d_param->set_negative_slope(negative_slope);
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, std::string filter, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases)
{
	return conv2d(input, filter, "", 0, pad_top_bottom, pad_left_right, vertical_filter_stride, horizontal_filter_stride, filter_height_dilation, filter_width_dialation, name, phases);
}

std::string DeepFlow::pooling(std::string input, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto pooling_param = node_param->mutable_pooling_param();
	pooling_param->set_h_pad(horizontalPadding);
	pooling_param->set_v_pad(verticalPadding);
	pooling_param->set_h_stride(horizontalStride);
	pooling_param->set_v_stride(verticalStride);
	pooling_param->set_window_h(windowHeight);
	pooling_param->set_window_w(windowWidth);	
	return node_param->output(0);
}

std::string DeepFlow::lifting(std::string input, LiftingMode mode, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto lifting_param = node_param->mutable_lifting_param();
	lifting_param->set_mode((deepflow::LiftingParam_Mode) mode);
	return node_param->output(0);
}

std::string DeepFlow::upsample(std::string input, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	node_param->mutable_upsample_param();	
	return node_param->output(0);
}

std::string DeepFlow::patching(std::string input, PatchingMode mode, int num_vertical_patches, int num_horizontal_patches, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto patching_param = node_param->mutable_patching_param();
	patching_param->set_mode((deepflow::PatchingParam_Mode) mode);
	patching_param->set_num_horizontal_patch(num_horizontal_patches);
	patching_param->set_num_vertical_patch(num_vertical_patches);
	return node_param->output(0);
}


std::string DeepFlow::_reduce(std::string input, int reduce_dimention, deepflow::ReduceParam_ReduceOp op, deepflow::ReduceParam::OutputType type, int output, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 2);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto reduce_param = node_param->mutable_reduce_param();
	reduce_param->set_reduce_dim(reduce_dimention);
	reduce_param->set_output_type(type);
	reduce_param->set_reduce_op(op);	
	return node_param->output(output);
}

std::string DeepFlow::_reduce_all(std::string input, ReduceAllOp op, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto reduce_param = node_param->mutable_reduce_all_param();
	reduce_param->set_reduce_op((deepflow::ReduceAllParam_ReduceAllOp) op);
	return node_param->output(0);
}

std::string DeepFlow::argmax(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_MAX, deepflow::ReduceParam::INDICES, 1, name, phases);
}

std::string DeepFlow::argmin(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_MIN, deepflow::ReduceParam::INDICES, 1, name, phases);
}

std::string DeepFlow::reduce_max(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_MAX, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_absmax(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_AMAX, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_min(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_MIN, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_norm1(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_NORM1, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_norm2(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_NORM2, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_mean(std::string input, std::string name, std::initializer_list<std::string> phases)
{
	return _reduce_all(input, DeepFlow::ReduceAllOp::REDUCE_ALL_AVG, name, phases);
}

std::string DeepFlow::reduce_sum(std::string input, std::string name, std::initializer_list<std::string> phases)
{
	return _reduce_all(input, DeepFlow::ReduceAllOp::REDUCE_ALL_SUM, name, phases);
}

std::string DeepFlow::reduce_sum(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_ADD, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_mean(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_AVG, deepflow::ReduceParam::VALUES, 0, name, phases);
}


std::string DeepFlow::phaseplexer(std::string input_1, std::string phase_1, std::string input_2, std::string phase_2, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input_1);
	node_param->add_input(input_2);
	auto phaseplexer_param = node_param->mutable_phaseplexer_param();
	phaseplexer_param->add_phase(phase_1);
	phaseplexer_param->add_phase(phase_2);	
	return node_param->output(0);
}

std::string DeepFlow::random_selector(std::string input_1, std::string input_2, float probability, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input_1);
	node_param->add_input(input_2);
	auto random_selector_param = node_param->mutable_random_selector_param();
	random_selector_param->set_probability(probability);	
	return node_param->output(0);
}

std::string DeepFlow::multiplexer(std::initializer_list<std::string> inputs, std::string selector, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);	
	for (auto phase : phases)
		node_param->add_phase(phase);
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);
	node_param->add_input(selector);
	auto multiplexer_param = node_param->mutable_multiplexer_param();
	multiplexer_param->set_num_inputs(inputs.size());
	return node_param->output(0);
}

std::string DeepFlow::loss(std::string a, std::string coef, ReduceOp op, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(coef);
	auto loss_param = node_param->mutable_loss_param();
	loss_param->set_reduce_op((deepflow::LossParam_ReduceOp)op);
	return node_param->output(0);
}

std::string DeepFlow::equal(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node_param();
	node_param->set_name(_block->get_unique_node_param_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
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

std::string DeepFlow::define_phase(std::string phase, PhaseBehaviour behaviour) {
	auto phase_param = _block->add_phase_param();	
	phase_param->set_phase(phase);
	phase_param->set_behaviour((deepflow::PhaseParam_PhaseBehaviour)behaviour);
	return phase;
}

std::string DeepFlow::define_train_phase(std::string train_phase)
{
	return define_phase(train_phase, TRAIN);
}

std::string DeepFlow::define_validation_phase(std::string validation_phase)
{
	return define_phase(validation_phase, VALIDATION);
}

std::string DeepFlow::define_inference_phase(std::string inference_phase)
{
	return define_phase(inference_phase, INFERENCE);
}

void DeepFlow::load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose)
{
	Caffe caffe(this, verbose);
	caffe.load(file_path, inputs);
}
