
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
DEFINE_string(run,"", "Phase to execute graph");
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
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.8, 1.0), 10000, "", "mnist_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.2), 10000, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	df.euclidean_loss(discriminator_input, labels_input);
	auto sub = df.subtract(discriminator_input, labels_input);
	auto reduce = df.reduce_norm1(sub, 0, "reduce");
	df.print({ reduce }, " NORM {0}\n", DeepFlow::EVERY_PASS); 
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

std::shared_ptr<Session> create_generator_session() {
	DeepFlow df;
	
	auto mean = 0;
	auto stddev = 0.01;
	auto negative_slope = 0.01;
	auto alpha_param = 1.0f;
	auto decay = 0.01f;
	auto dropout = 0.4;	
	int depth = 512;

	auto g_solver = df.adam_solver(0.0002f, 0.5f, 0.999f);
	auto gin = df.data_generator(df.random_uniform({ FLAGS_batch, 100, 1, 1 }, -1, 1, "random_input"), 10000, "", "input");
	
	auto gfc_w = df.variable(df.random_normal({ 100, depth, 4, 4 }, mean, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");

	auto gconv1_f = df.variable(df.random_normal({ depth, depth/2, 2, 2 }, mean, stddev), g_solver, "gconv1_f");
	auto gconv1_t = df.transposed_conv2d(gfc, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1");
	auto gconv1_n = df.batch_normalization(gconv1_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);
	auto gconv1_r = df.leaky_relu(gconv1_n, negative_slope);	
	auto gconv1_d = df.dropout(gconv1_r, dropout);

	auto gconv2_f = df.variable(df.random_normal({ depth/2, depth/4, 5, 5 }, mean, stddev), g_solver, "gconv2_f");
	auto gconv2_t = df.transposed_conv2d(gconv1_d, gconv2_f, 2, 2, 2, 2, 1, 1, "gconv2");
	auto gconv2_n = df.batch_normalization(gconv2_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);
	auto gconv2_r = df.leaky_relu(gconv2_n, negative_slope);
	auto gconv2_d = df.dropout(gconv2_r, dropout);

	auto gconv3_f = df.variable(df.random_normal({ depth/4, 1, 5, 5 }, mean, stddev), g_solver, "gconv3_f");
	auto gconv3_t = df.transposed_conv2d(gconv2_d, gconv3_f, 2, 2, 2, 2, 1, 1, "gconv3");	
	auto gconv3_n = df.batch_normalization(gconv3_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);

	auto gout = df.tanh(gconv3_n, "gout");
	auto disp = df.display(gout, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 10);
	
	//auto replay_memory = df.replay_memory(gout, 10000, "replay_memory");
	
	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.1;
	auto d_solver = df.adam_solver(0.0002f, 0.5f, 0.999f);
	auto negative_slope = 0.1;
	auto dropout = 0.2;
	auto depth = 64;

	auto input = df.place_holder({ FLAGS_batch , 1, 28, 28 }, Tensor::Float, "input");		

	auto conv1_w = df.variable(df.random_uniform({ depth, 1, 5, 5 }, -0.100000, 0.100000), d_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 2, 2, 2, 2, 1, 1, "conv1");	
	auto conv1_r = df.leaky_relu(conv1, negative_slope);	
	
	auto conv2_w = df.variable(df.random_uniform({ depth*2, depth, 3, 3 }, -0.100000, 0.100000), d_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_r, conv2_w, 1, 1, 2, 2, 1, 1, "conv2");	
	auto conv2_r = df.leaky_relu(conv2, negative_slope);	

	auto conv3_w = df.variable(df.random_uniform({ depth*2, depth*2, 3, 3 }, -0.100000, 0.100000), d_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_r, conv3_w, 1, 1, 2, 2, 1, 1, "conv3");
	auto conv3_r = df.leaky_relu(conv3, negative_slope);	
	auto drop1 = df.dropout(conv3_r, dropout);

	auto w1 = df.variable(df.random_uniform({ depth*2*4*4, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1");
	auto m1 = df.matmul(drop1, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");
	auto drop2 = df.dropout(relu1, dropout);

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

	auto generator = create_generator_session();
	generator->initialize();
	generator->set_execution_context(execution_context);
	auto discriminator = create_discriminator();	
	discriminator->initialize();
	discriminator->set_execution_context(execution_context);
	auto mnist_reader = create_mnist_reader();
	mnist_reader->initialize();
	mnist_reader->set_execution_context(execution_context);
	auto mnist_labels = create_mnist_labels();
	mnist_labels->initialize();
	mnist_labels->set_execution_context(execution_context);
	auto generator_labels = create_generator_labels();
	generator_labels->initialize();
	generator_labels->set_execution_context(execution_context);
	auto loss = create_loss();
	loss->initialize();
	loss->set_execution_context(execution_context);
	auto socket_io_sender_session = create_socket_io_sender();
	socket_io_sender_session->set_execution_context(execution_context);
	socket_io_sender_session->initialize();

	auto mnist_reader_data = mnist_reader->get_node("mnist_data");
	auto generator_output = generator->get_node("gout");
	//auto replay_memory_output = generator->get_node("replay_memory");
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->get_node("dout");
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto mnist_labels_output = mnist_labels->get_node("mnist_labels");
	auto loss_discriminator_input = loss->get_placeholder("discriminator_input");
	auto loss_labels_input = loss->get_placeholder("labels_input");
	auto loss_output = loss->get_node("reduce");
	auto sio_sender_dtrue = socket_io_sender_session->get_node("dtrue");
	auto sio_sender_dfake = socket_io_sender_session->get_node("dfake");
	auto sio_sender_gfake = socket_io_sender_session->get_node("gfake");

	for (int i = 1; i <= FLAGS_epoch && execution_context->quit != true; ++i) {
		
		execution_context->current_epoch = i;

		std::cout << "Epoch: " << i << std::endl;		
		
		std::cout << " MNIST INPUT " << std::endl;
		discriminator->reset_gradients();
		loss->reset_gradients();		
		mnist_reader_data->forward();
		discriminator_input->feed_forward(mnist_reader_data, 0);
		discriminator->forward();
		mnist_labels->forward();
		loss_discriminator_input->feed_forward(discriminator_output, 0);
		loss_labels_input->feed_forward(mnist_labels_output, 0);
		loss->forward();
		sio_sender_dtrue->feed_forward(loss_output, 0);
		loss->backward();
		discriminator_output->feed_backward(loss_discriminator_input, 0);
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " GENERATOR INPUT " << std::endl;
		discriminator->reset_gradients();
		loss->reset_gradients();		
		generator->forward();
		discriminator_input->feed_forward(generator_output, 0);
		discriminator->forward();
		generator_labels->forward();
		loss_discriminator_input->feed_forward(discriminator_output, 0);
		loss_labels_input->feed_forward(generator_labels_output, 0);
		loss->forward();
		sio_sender_dfake->feed_forward(loss_output, 0);
		loss->backward();
		discriminator_output->feed_backward(loss_discriminator_input, 0);
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " TRAINING GENERATOR " << std::endl;
		generator->reset_gradients();
		discriminator->reset_gradients();
		loss->reset_gradients();
		generator->forward();
		discriminator_input->feed_forward(generator_output, 0);
		discriminator->forward();
		mnist_labels->forward();		
		loss_discriminator_input->feed_forward(discriminator_output, 0);
		loss_labels_input->feed_forward(mnist_labels_output, 0);
		loss->forward();
		sio_sender_gfake->feed_forward(loss_output, 0);		
		loss->backward();
		discriminator_output->feed_backward(loss_discriminator_input, 0);
		discriminator->backward();
		generator_output->feed_backward(discriminator_input, 0);
		generator->backward();
		generator->apply_solvers();

		socket_io_sender_session->forward();
	}

}

