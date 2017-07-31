
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000000, "Maximum epochs");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save, 0 , "Save model epoch frequency (Don't Save = 0)");

std::shared_ptr<Session> load_generator_session(std::string suffix) {
	std::string filename = "g" + suffix + ".bin";
	std::cout << "Loading generator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> load_disc_session(std::string suffix) {
	std::string filename = "d" + suffix + ".bin";
	std::cout << "Loading discriminator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> create_mnist_reader() {
	DeepFlow df;
	df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");
	return df.session();
}

std::shared_ptr<Session> create_mnist_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.75, 1.0), 10000, "", "mnist_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.25), 10000, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto euc = df.square_error(discriminator_input, labels_input);
	auto loss = df.loss(euc, DeepFlow::SUM, "loss");
	df.print({ loss }, " NORM {0}\n", DeepFlow::EVERY_PASS); 
	return df.session();
}

std::shared_ptr<Session> create_socket_io_sender() {
	DeepFlow df;
	auto dtrue = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "dtrue");
	auto dfake = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "dfake");
	auto gfake = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "gfake");
	df.sio_output({ dtrue, dfake, gfake });
	return df.session();
}

std::shared_ptr<Session> create_display_session() {
	DeepFlow df;
	auto input = df.place_holder({ 100, 1, 28, 28 }, Tensor::Float, "input");
	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp");
	return df.session();
}

std::shared_ptr<Session> create_generator_session() {
	DeepFlow df;

	auto mean = 0;
	auto stddev = 0.1;
	auto negative_slope = 0;
	auto alpha_param = 0.05;
	auto dropout = 0.3;
	auto exp_avg_factor = 0.1;
	int depth = 1024;

	auto g_solver = df.adam_solver(0.0002f, 0.5f, 0.9999f);
	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, mean, stddev, "random_input"), 10000, "", "input");

	auto gfc_w = df.variable(df.random_normal({ 100, depth, 4, 4 }, mean, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");
	auto gfc_r = df.leaky_relu(gfc, negative_slope);
	auto gfc_drop = df.dropout(gfc_r, dropout);

	auto gconv1_f = df.variable(df.random_normal({ depth, depth / 2, 2, 2 }, mean, stddev), g_solver, "gconv1_f");
	auto gconv1_t = df.transposed_conv2d(gfc_drop, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1");
	auto gconv1_n = df.batch_normalization(gconv1_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	auto gconv1_r = df.leaky_relu(gconv1_n, negative_slope);
	auto gconv1_d = df.dropout(gconv1_r, dropout);

	auto gconv2_f = df.variable(df.random_normal({ depth / 2, depth / 4, 5, 5 }, mean, stddev), g_solver, "gconv2_f");
	auto gconv2_t = df.transposed_conv2d(gconv1_d, gconv2_f, 2, 2, 2, 2, 1, 1, "gconv2");
	auto gconv2_n = df.batch_normalization(gconv2_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);
	auto gconv2_r = df.leaky_relu(gconv2_n, negative_slope);
	auto gconv2_d = df.dropout(gconv2_r, dropout);

	auto gconv3_f = df.variable(df.random_normal({ depth / 4, 1, 5, 5 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv3_t = df.transposed_conv2d(gconv2_d, gconv3_f, 2, 2, 2, 2, 1, 1, "gconv3");
	auto gconv3_n = df.batch_normalization(gconv3_t, DeepFlow::SPATIAL, exp_avg_factor, 1, 0, alpha_param, 1);

	auto gout = df.tanh(gconv3_n, "gout");	

	//auto replay_memory = df.replay_memory(gout, 10000, "replay_memory");

	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.02;
	auto d_solver = df.adam_solver(0.0002f, 0.9f, 0.999f);
	auto negative_slope = 0.1;
	auto input = df.place_holder({ FLAGS_batch , 1, 28, 28 }, Tensor::Float, "input");

	auto conv1_w = df.variable(df.random_uniform({ 64, 1, 3, 3 }, -0.100000, 0.100000), d_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 1, 1, 2, 2, 1, 1, "conv1");
	auto conv1_r = df.leaky_relu(conv1, negative_slope);
	//auto conv1_n = df.batch_normalization(conv1_r, DeepFlow::SPATIAL);

	auto conv2_w = df.variable(df.random_uniform({ 64, 64, 3, 3 }, -0.100000, 0.100000), d_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_r, conv2_w, 1, 1, 2, 2, 1, 1, "conv2");
	auto conv2_r = df.leaky_relu(conv2, negative_slope);
	//auto conv2_n = df.batch_normalization(conv2_r, DeepFlow::SPATIAL);

	auto conv3_w = df.variable(df.random_uniform({ 128, 64, 3, 3 }, -0.100000, 0.100000), d_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_r, conv3_w, 1, 1, 2, 2, 1, 1, "conv3");
	auto conv3_r = df.leaky_relu(conv3, negative_slope);
	//auto conv3_n = df.batch_normalization(conv3_r, DeepFlow::SPATIAL);			
	auto drop1 = df.dropout(conv3_r);

	auto w1 = df.variable(df.random_uniform({ 2048, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1");
	auto m1 = df.matmul(drop1, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");
	auto drop2 = df.dropout(relu1);

	auto w2 = df.variable(df.random_uniform({ 500, 1, 1, 1 }, -0.100000, 0.100000), d_solver, "w2");
	auto m2 = df.matmul(drop2, w2, "m2");
	auto b2 = df.variable(df.step({ 1, 1, 1, 1 }, -1.000000, 1.000000), d_solver, "b2");
	auto bias2 = df.bias_add(m2, b2, "bias2");

	auto dout = df.sigmoid(bias2, "dout");

	//df.logger({ conv1_r, conv1_n}, "./log.txt", "R -> {0}\nN -> {0}\n\n", DeepFlow::EVERY_PASS);

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> generator;
	if (FLAGS_load.empty())
		generator = create_generator_session();
	else
		generator = load_generator_session(FLAGS_load);			
	generator->initialize(execution_context);	

	std::shared_ptr<Session> discriminator;

	if (FLAGS_load.empty())
		discriminator = create_discriminator();
	else
		discriminator = load_disc_session(FLAGS_load);		
	discriminator->initialize(execution_context);	

	auto mnist_reader = create_mnist_reader();
	mnist_reader->initialize(execution_context);	

	auto mnist_labels = create_mnist_labels();
	mnist_labels->initialize(execution_context);	

	auto generator_labels = create_generator_labels();
	generator_labels->initialize(execution_context);	

	auto loss = create_loss();
	loss->initialize(execution_context);	

	auto socket_io_sender_session = create_socket_io_sender();
	socket_io_sender_session->initialize(execution_context);

	auto disp_session = create_display_session();
	disp_session->initialize(execution_context);
	

	auto mnist_reader_data = mnist_reader->get_node("mnist_data");
	auto generator_output = generator->get_node("gout");	
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->get_node("dout");
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto mnist_labels_output = mnist_labels->get_node("mnist_labels");
	auto loss_discriminator_input = loss->get_placeholder("discriminator_input");
	auto loss_labels_input = loss->get_placeholder("labels_input");
	auto loss_output = loss->get_node("loss");
	auto sio_sender_dtrue = socket_io_sender_session->get_node("dtrue");
	auto sio_sender_dfake = socket_io_sender_session->get_node("dfake");
	auto sio_sender_gfake = socket_io_sender_session->get_node("gfake");
	auto disp = disp_session->get_node("input");

	int iter;
	for (iter = 1; iter <= FLAGS_epoch && execution_context->quit != true; ++iter) {
		
		execution_context->current_epoch = iter;

		std::cout << "Epoch: " << iter << std::endl;		
		
		std::cout << " MNIST INPUT " << std::endl;
		mnist_reader_data->forward();
		discriminator_input->write_values(mnist_reader_data->output(0)->value());
		discriminator->forward();
		mnist_labels->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(mnist_labels_output->output(0)->value());
		loss->forward();
		sio_sender_dtrue->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " GENERATOR INPUT " << std::endl;
		generator->forward();
		disp->write_values(generator_output->output(0)->value());
		disp_session->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		generator_labels->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(generator_labels_output->output(0)->value());
		loss->forward();
		sio_sender_dfake->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers(); 

		std::cout << " TRAINING GENERATOR " << std::endl;
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		mnist_labels->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(mnist_labels_output->output(0)->value());
		loss->forward();
		sio_sender_gfake->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator->reset_gradients();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();

		socket_io_sender_session->forward();

		if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
			discriminator->save("d" + std::to_string(iter) + ".bin");
			generator->save("g" + std::to_string(iter) + ".bin");
		}
	}

	if (FLAGS_save != 0) {
		discriminator->save("d" + std::to_string(iter) + ".bin");
		generator->save("g" + std::to_string(iter) + ".bin");
	}

}

