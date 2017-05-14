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

std::string DeepFlow::mnist_reader(std::string folder_path, int batch_size, MNISTReader::MNISTReaderType reader_type, MNISTReader::MNISTOutputType output_type, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto generator_param = node_param->mutable_generator_param();
	auto mnistParam = generator_param->mutable_mnist_param();
	mnistParam->set_folder_path(folder_path);
	mnistParam->set_reader_type((deepflow::MnistParam::ReaderType) reader_type);
	mnistParam->set_output_type((deepflow::MnistParam::OutputType) output_type);
	mnistParam->set_batch_size(batch_size);
	return node_param->output(0);

}

std::string DeepFlow::data_generator(std::string initializer, int num_samples, std::string solver, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto variable_param = node_param->mutable_variable_param();	
	auto generator_param = node_param->mutable_generator_param();
	auto data_generator_param = generator_param->mutable_data_generator_param();
	data_generator_param->set_num_samples(num_samples);
	if (!solver.empty())
		variable_param->set_solver_name(solver);
	auto init_param = variable_param->mutable_init_param();
	init_param->CopyFrom(*_block->find_initializer_by_name(initializer));	
	return node_param->output(0);
}

std::string DeepFlow::image_reader(std::string file_path, deepflow::ImageReaderParam_Type type, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto generator_param = node_param->mutable_generator_param();
	auto image_reader_param = generator_param->mutable_image_reader_param();
	image_reader_param->set_file_name(file_path);
	image_reader_param->set_type(type);	
	return node_param->output(0);
}

std::string DeepFlow::image_batch_reader(std::string folder_path, std::initializer_list<int> dims, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto generator_param = node_param->mutable_generator_param();
	auto image_batch_reader_param = generator_param->mutable_image_batch_reader_param();
	image_batch_reader_param->set_folder_path(folder_path);	

	std::vector<int> values(dims);
	auto tensor_param = image_batch_reader_param->mutable_tensor_param();
	for (int i = 0; i < values.size(); ++i)
		tensor_param->add_dims(values[i]);
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);	
	return node_param->output(0);
}

std::string DeepFlow::add(std::string a, std::string b, float alpha, float beta, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
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

std::string DeepFlow::subtract(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	return add(a, b, 1.0f, -1.0f, name, phases);
}

std::string DeepFlow::bias_add(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto bias_add_param = node_param->mutable_bias_add_param();	
	return node_param->output(0);
}

std::string DeepFlow::index_fill(std::initializer_list<int> dims, float offset, std::string name) {
	auto init_param = _block->add_initializer();
	init_param->set_name(_block->get_unique_initializer_name(name));
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
	auto init_param = _block->add_initializer();
	init_param->set_name(_block->get_unique_initializer_name(name));
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

std::string DeepFlow::fill(std::initializer_list<int> dims, float value, std::string name) {
	auto init_param = _block->add_initializer();
	init_param->set_name(_block->get_unique_initializer_name(name));
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
	auto init_param = _block->add_initializer();
	init_param->set_name(_block->get_unique_initializer_name(name));
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
	auto init_param = _block->add_initializer();
	init_param->set_name(_block->get_unique_initializer_name(name));
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
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);	
	auto bias_add_param = node_param->mutable_bias_add_param();	
	return node_param->output(0);
}

std::string DeepFlow::softmax(std::string a, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto softmax_param = node_param->mutable_softmax_param();
	softmax_param->set_alpha(1.0f);
	softmax_param->set_beta(0.0f);	
	return node_param->output(0);
}

std::string DeepFlow::square(std::string a, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto square_param = node_param->mutable_square_param();
	return node_param->output(0);
}

std::string DeepFlow::place_holder(std::array<int, 4> dims, Tensor::TensorType type, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
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

std::string DeepFlow::variable(std::string initializer, std::string solver, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	auto variable_param = node_param->mutable_variable_param();
	if (!solver.empty())
		variable_param->set_solver_name(solver);
	auto init_param = variable_param->mutable_init_param();
	init_param->CopyFrom(*_block->find_initializer_by_name(initializer));
	return node_param->output(0);
}

std::string DeepFlow::matmul(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto matmul_param = node_param->mutable_matmul_param();
	matmul_param->set_alpha(1.0f);
	matmul_param->set_beta(0.0f);	
	return node_param->output(0);
}

std::string DeepFlow::leaky_relu(std::string a, float negative_slope, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);	
	auto relu_param = node_param->mutable_leaky_relu_param();
	relu_param->set_negative_slope(negative_slope);	
	return node_param->output(0);
}

std::string DeepFlow::sigmoid(std::string a, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
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
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
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
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
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
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
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
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto activation_param = node_param->mutable_activation_param();
	activation_param->set_coef(alpha);
	activation_param->set_type(deepflow::ActivationParam_Type_CUDNN_ACTIVATION_ELU);	
	return node_param->output(0);	
}

std::array<std::string, 2> DeepFlow::accumulator(std::string input, Accumulator::ResetTime resetTime, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 2);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);	
	auto accumulator_param = node_param->mutable_accumulator_param();
	accumulator_param->set_reset_time((deepflow::AccumulatorParam_ResetTime)resetTime);
	std::array<std::string, 2> outputs;
	outputs[0] = node_param->output(0);
	outputs[1] = node_param->output(1);
	return outputs;	
}

void DeepFlow::print(std::initializer_list<std::string> inputs, std::string message, Print::PrintTime printTime, Print::PrintType printType, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	std::vector<std::string> inputsVec(inputs.size());
	std::copy(inputs.begin(), inputs.end(), inputsVec.begin());
	for (int i = 0; i < inputsVec.size(); ++i)
		node_param->add_input(inputsVec[i]);	
	auto print_param = node_param->mutable_print_param();	
	print_param->set_message(message);
	print_param->set_num_inputs(inputs.size());
	print_param->set_print_time((deepflow::PrintParam_PrintTime)printTime);
	print_param->set_print_type((deepflow::PrintParam_PrintType)printType);	
}

std::string DeepFlow::softmax_loss(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto loss_param = node_param->mutable_loss_param();
	auto softmax_loss_param = loss_param->mutable_softmax_loss_param();
	softmax_loss_param->set_alpha(1.0f);
	softmax_loss_param->set_beta(0.0f);
	return node_param->output(0);	
}

void DeepFlow::euclidean_loss(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();	
	node_param->set_name(_block->get_unique_node_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto loss_param = node_param->mutable_loss_param();
	auto euclidean_loss_param = loss_param->mutable_euclidean_loss_param();	
}

std::string DeepFlow::sgd_solver(float momentum, float learning_rate, std::string name) {
	auto solver_param = _block->add_solver();	
	solver_param->set_name(_block->get_unique_solver_name(name));
	auto sgd_solver = solver_param->mutable_sgd_solver();
	sgd_solver->set_learning_rate(learning_rate);
	sgd_solver->set_momentum(momentum);
	return solver_param->name();
}

std::string DeepFlow::gain_solver(float momentum, float learning_rate, float max_gain, float min_gain, float gain_plus, float gain_mult, std::string name) {
	auto solver_param = _block->add_solver();	
	solver_param->set_name(_block->get_unique_solver_name(name));
	auto gain_solver = solver_param->mutable_gain_solver();
	gain_solver->set_momentum(momentum);
	gain_solver->set_learning_rate(learning_rate);
	gain_solver->set_max_gain(max_gain);
	gain_solver->set_min_gain(min_gain);
	gain_solver->set_gain_plus(gain_plus);
	gain_solver->set_gain_mult(gain_mult);
	return solver_param->name();
}
std::string DeepFlow::adam_solver(float learning_rate, float beta1, float beta2, float eps, std::string name) {	
	auto solver_param = _block->add_solver();
	solver_param->set_name(_block->get_unique_solver_name(name));
	auto adam_solver = solver_param->mutable_adam_solver();	
	adam_solver->set_learning_rate(learning_rate);
	adam_solver->set_beta1(beta1);
	adam_solver->set_beta2(beta2);
	adam_solver->set_eps(eps);
	return solver_param->name();
}

std::string DeepFlow::adadelta_solver(float learning_rate, float momentum, float delta, std::string name) {
	auto solver_param = _block->add_solver();
	solver_param->set_name(_block->get_unique_solver_name(name));
	auto adadelta_solver = solver_param->mutable_adadelta_solver();
	adadelta_solver->set_learning_rate(learning_rate);
	adadelta_solver->set_momentum(momentum);
	adadelta_solver->set_delta(delta);	
	return solver_param->name();
}

std::string DeepFlow::dropout(std::string a, float dropout, bool train_only, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	auto dropout_param = node_param->mutable_dropout_param();	
	dropout_param->set_dropout(dropout);
	dropout_param->set_train_only(train_only);	
	return node_param->output(0);
}

std::string DeepFlow::transposed_conv2d(std::string input, std::string filter,std::array<int, 4> dims, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	node_param->add_input(filter);
	auto tconv_2d_param = node_param->mutable_transposed_conv_2d_param();
	auto tensor_param = tconv_2d_param->mutable_tensor_param();
	tensor_param->set_type(deepflow::TensorParam_TensorType_FLOAT);
	for (int i = 0; i < 4; ++i)
		tensor_param->add_dims(dims[i]);
	tconv_2d_param->set_pad_h(pad_top_bottom);
	tconv_2d_param->set_pad_w(pad_left_right);
	tconv_2d_param->set_u(vertical_filter_stride);
	tconv_2d_param->set_v(horizontal_filter_stride);
	tconv_2d_param->set_dilation_h(filter_height_dilation);
	tconv_2d_param->set_dilation_w(filter_width_dialation);	
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, std::string filter, std::string bias, int pad_top_bottom, int pad_left_right, int vertical_filter_stride, int horizontal_filter_stride, int filter_height_dilation, int filter_width_dialation, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
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
	return node_param->output(0);
}

std::string DeepFlow::conv2d(std::string input, std::string filter, std::string bias, std::string name, std::initializer_list<std::string> phases) {
	return conv2d(input, filter, bias, 0, 0, 1, 1, 1, 1, name, phases);
}

void DeepFlow::display(std::string input,int delay_msec, deepflow::DisplayParam_DisplayType type, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input);
	auto display_param = node_param->mutable_display_param();
	display_param->set_delay_msec(delay_msec);
	display_param->set_display_type(type);	
}

void DeepFlow::psnr(std::string a, std::string b, Psnr::PrintTime printTime, std::string name, std::initializer_list<std::string> phases)
{
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto psnr_param = node_param->mutable_psnr_param();
	psnr_param->set_print_time((deepflow::PsnrParam_PrintTime)printTime);	
}

std::string DeepFlow::pooling(std::string input, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
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

std::string DeepFlow::_reduce(std::string input, int reduce_dimention, deepflow::ReduceParam_ReduceOp op, deepflow::ReduceParam::OutputType type, int output, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
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

std::string DeepFlow::reduce_sum(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_ADD, deepflow::ReduceParam::VALUES, 0, name, phases);
}

std::string DeepFlow::reduce_mean(std::string input, int reduceDimension, std::string name, std::initializer_list<std::string> phases) {
	return _reduce(input, reduceDimension, deepflow::ReduceParam_ReduceOp_AVG, deepflow::ReduceParam::VALUES, 0, name, phases);
}


std::string DeepFlow::phaseplexer(std::string input_1, std::string phase_1, std::string input_2, std::string phase_2, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
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
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(input_1);
	node_param->add_input(input_2);
	auto random_selector_param = node_param->mutable_random_selector_param();
	random_selector_param->set_probability(probability);	
	return node_param->output(0);
}

std::string DeepFlow::equal(std::string a, std::string b, std::string name, std::initializer_list<std::string> phases) {
	auto node_param = _block->add_node();
	node_param->set_name(_block->get_unique_node_name(name));
	add_outputs(node_param, 1);
	for (auto phase : phases)
		node_param->add_phase(phase);
	node_param->add_input(a);
	node_param->add_input(b);
	auto equal_param = node_param->mutable_equal_param();	
	return node_param->output(0);
}

std::shared_ptr<Block> DeepFlow::block() {
	return _block;
}

std::shared_ptr<Session> DeepFlow::session()
{	
	auto session = std::make_shared<Session>();
	session->setBlock(_block);
	return session;
}

void DeepFlow::define_phase(std::string phase, deepflow::PhaseParam_PhaseBehaviour behaviour) {
	auto phase_param = _block->add_phase();	
	phase_param->set_phase(phase);
	phase_param->set_behaviour(behaviour);	
}

void DeepFlow::define_train_phase(std::string train_phase)
{
	define_phase(train_phase, deepflow::PhaseParam_PhaseBehaviour_TRAIN);
}

void DeepFlow::define_validation_phase(std::string validation_phase)
{
	define_phase(validation_phase, deepflow::PhaseParam_PhaseBehaviour_VALIDATION);
}

void DeepFlow::define_inference_phase(std::string inference_phase)
{
	define_phase(inference_phase, deepflow::PhaseParam_PhaseBehaviour_INFERENCE);
}

void DeepFlow::load_from_caffe_model(std::string file_path, std::initializer_list<std::pair<std::string, std::array<int, 4>>> inputs, bool verbose)
{
	Caffe caffe(this, verbose);
	caffe.load(file_path, inputs);
}
