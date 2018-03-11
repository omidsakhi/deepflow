#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_string(run, "", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");

std::shared_ptr<Session> create_mnist_reader() {
	DeepFlow df;
	df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");
	return df.session();
}

std::shared_ptr<Session> create_mnist_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.8, 1.0), "", "mnist_labels");
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.2), "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto euc = df.square_error(discriminator_input, labels_input);
	auto loss = df.loss(euc, DeepFlow::AVG, "loss");
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

std::string tconv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, int kernel, int pad, bool activation) {
	auto mean = 0;
	auto stddev = 0.001;
	auto dropout = 0.2;	
	auto tconv_l = df->resize(input, 2.0, 2.0, name + "_resize");
	auto tconv_f = df->variable(df->random_normal({ depth1, depth2, kernel, kernel }, mean, stddev), solver, name + "_f");
	auto tconv_t = df->conv2d(tconv_l, tconv_f, pad, pad, 1, 1, 1, 1, name + "_t");
	//auto tconv_b = df->batch_normalization(tconv_t, DeepFlow::SPATIAL, 0, 1.0f, 0.0f, 0.15f, 1.0f, name + "_b");
	if (activation) {
		auto drop = df->dropout(tconv_t, dropout);
		return df->relu(drop);
	}
	return tconv_t;
}

std::shared_ptr<Session> create_generator_session() {
	DeepFlow df;

	auto mean = 0;
	auto stddev = 0.1;	
	auto dropout = 0.2;	

	auto g_solver = df.adam_solver(0.0002f, 0.5f, 0.999f);
	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1, "random_input"), "", "input");

	auto gfc_w = df.variable(df.random_normal({ 100, 512, 3, 3 }, mean, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");
	auto gfc_r = df.relu(gfc);
	auto gfc_drop = df.dropout(gfc_r, dropout);

	auto tconv1 = tconv(&df, "tconv1", gfc_drop, g_solver, 256, 512, 2, 0, true);
	auto tconv2 = tconv(&df, "tconv2", tconv1, g_solver, 128, 256, 2, 0, true);
	auto tconv3 = tconv(&df, "tconv3", tconv2, g_solver, 64, 128, 3, 0, true);
	auto tconv4 = tconv(&df, "tconv4", tconv3, g_solver, 1, 64, 5, 0, false);

	auto gout = df.tanh(tconv4, "gout");
	auto disp = df.display(gout, 500, DeepFlow::EVERY_PASS, DeepFlow::VALUES);

	//auto replay_memory = df.replay_memory(gout, 10000, "replay_memory");

	return df.session();
}

std::string conv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, bool activation) {
	auto mean = 0;
	auto stddev = 0.1;
	auto negative_slope = 0.1;	
	auto conv_w = df->variable(df->random_normal({ depth1, depth2, 3, 3 }, mean, stddev), solver, name + "_w");
	auto conv_ = df->conv2d(input, conv_w, 1, 1, 2, 2, 1, 1, name);
	if (activation) {		
		auto drop = df->dropout(conv_, 0.2);
		return df->leaky_relu(drop, negative_slope);
	}
	return conv_;
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.01;
	auto d_solver = df.adam_solver(0.0002f, 0.5f, 0.999f);
	auto negative_slope = 0.1;
	auto input = df.place_holder({ FLAGS_batch , 1, 28, 28 }, Tensor::Float, "input");

	//auto disp = df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES);

	auto conv1 = conv(&df, "conv1", input, d_solver, 64, 1, true);
	auto conv2 = conv(&df, "conv2", conv1, d_solver, 64, 64, true);
	auto conv3 = conv(&df, "conv3", conv2, d_solver, 64, 64, true);		
	auto conv4 = conv(&df, "conv4", conv2, d_solver, 128, 64, true);

	auto w1 = df.variable(df.random_uniform({ 2048, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1");
	auto m1 = df.matmul(conv4, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");
	auto drop2 = df.dropout(relu1, 0.2);

	auto w2 = df.variable(df.random_uniform({ 500, 1, 1, 1 }, -0.100000, 0.100000), d_solver, "w2");
	auto m2 = df.matmul(drop2, w2, "m2");

	auto dout = df.sigmoid(m2, "dout");

	//df.logger({ conv1_r, conv1_n}, "./log.txt", "R -> {0}\nN -> {0}\n\n", DeepFlow::EVERY_PASS);

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	auto generator = create_generator_session();
	generator->initialize();
	generator->set_execution_context(execution_context);
	auto discriminator = create_discriminator();
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

	auto mnist_reader_data = mnist_reader->get_node("mnist_data");
	auto generator_output = generator->get_node("gout");
	//auto replay_memory_output = generator->get_node("replay_memory");
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

	for (int i = 1; i <= FLAGS_epoch && execution_context->quit != true; ++i) {
		std::cout << "Epoch: " << i << std::endl;

		std::cout << " MNIST INPUT " << std::endl;
		mnist_reader_data->forward();
		discriminator_input->write_values(mnist_reader_data->output(0)->value());		
		discriminator->forward();
		mnist_labels->forward();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());		
		loss_labels_input->write_values(mnist_labels_output->output(0)->value());		
		loss->forward();
		sio_sender_dtrue->write_values(loss_output->output(0)->value());
		loss->backward();		
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " GENERATOR INPUT " << std::endl;
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		generator_labels->forward();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(generator_labels_output->output(0)->value());
		loss->forward();
		sio_sender_dfake->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " TRAINING GENERATOR " << std::endl;
		loss->reset_gradients();
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		mnist_labels->forward();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(mnist_labels_output->output(0)->value());
		loss->forward();
		sio_sender_gfake->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();

		socket_io_sender_session->forward();
	}

}