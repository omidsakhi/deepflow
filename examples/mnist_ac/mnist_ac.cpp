
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
DEFINE_int32(batch, 100, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 10000, "Maximum epochs");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save, 0 , "Save model epoch frequency (Don't Save = 0)");
DEFINE_bool(train, false, "Train");

std::shared_ptr<Session> load_session(std::string suffix) {
	std::string filename = "mnist_ac" + suffix + ".bin";
	std::cout << "Loading generator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> create_mnist_session() {
	DeepFlow df;

	auto input = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");

	return df.session();
}

std::shared_ptr<Session> create_noise_session() {
	DeepFlow df;
	
	auto solver = df.sgd_solver(0.99f, 0.00001f);

	auto noise = df.data_generator(df.random_normal({ 100, 1, 28, 28 }, 0, -1), 60000, solver, "noise");

	return df.session();
}

std::shared_ptr<Session> create_autoencoder_session() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.02;
	auto enc_solver = df.adam_solver(0.0002f, 0.5f);
	auto dec_solver = df.adam_solver(0.0002f, 0.5f);
	auto enc_negative_slope = 0.05;
	auto dec_negative_slope = 0.05;		
	auto alpha_param = 1.0f;
	auto decay = 0.01f;
	auto depth = 256;

	

	auto input = df.place_holder({ 100, 1, 28, 28 }, Tensor::Float, "input");

	auto conv1_w = df.variable(df.random_normal({ depth, 1, 3, 3 }, mean, stddev), enc_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 1, 1, 1, 1, 1, 1, "conv1");	
	auto conv1_r = df.leaky_relu(conv1, enc_negative_slope);
	auto conv1_p = df.pooling(conv1_r, 2, 2, 0, 0, 2, 2, "conv1_p");
	
	auto conv2_w = df.variable(df.random_normal({ depth, depth, 3, 3 }, mean, stddev), enc_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_p, conv2_w, 1, 1, 1, 1, 1, 1, "conv2");		
	auto conv2_r = df.leaky_relu(conv2, enc_negative_slope);
	auto conv2_p = df.pooling(conv2_r, 2, 2, 0, 0, 2, 2, "conv2_p");

	auto conv3_w = df.variable(df.random_normal({ 8, depth, 3, 3 }, mean, stddev), enc_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_p, conv3_w, 1, 1, 1, 1, 1, 1, "conv3");	
	auto conv3_r = df.leaky_relu(conv3, enc_negative_slope);
	auto conv3_p = df.pooling(conv3_r, 2, 2, 1, 1, 2, 2, "conv3_p");

	auto tconv1_f = df.variable(df.random_normal({ 8, depth, 2, 2 }, mean, stddev), dec_solver, "tconv1_f");
	auto tconv1_t = df.transposed_conv2d(conv3_p, tconv1_f, 1, 1, 2, 2, 1, 1, "tconv1_t");
	auto tconv1_b = df.batch_normalization(tconv1_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);
	auto tconv1_r = df.leaky_relu(tconv1_b, dec_negative_slope);

	auto tconv2_f = df.variable(df.random_normal({ depth, depth, 3, 3 }, mean, stddev), dec_solver, "tconv2_f");
	auto tconv2_t = df.transposed_conv2d(tconv1_r, tconv2_f, 1, 1, 2, 2, 1, 1, "tconv2_t");
	auto tconv2_b = df.batch_normalization(tconv2_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);
	auto tconv2_r = df.leaky_relu(tconv2_b, dec_negative_slope);

	auto tconv3_f = df.variable(df.random_normal({ depth, 1, 3, 3 }, mean, stddev), dec_solver, "tconv3_f");
	auto tconv3_t = df.transposed_conv2d(tconv2_r, tconv3_f, 1, 1, 2, 2, 1, 1, "tconv3_t");

	auto output = df.tanh(tconv3_t, "output");	
	
	df.display(output, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 10, "disp");

	auto sqerr = df.square_error(output, input);
	auto loss = df.loss(sqerr, DeepFlow::AVG);	
	
	df.print({ loss }, "Epoch: {ep} - Loss: {0}\n", DeepFlow::EVERY_PASS);

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> autoencoder_session;
	if (FLAGS_load.empty())
		autoencoder_session = create_autoencoder_session();
	else
		autoencoder_session = load_session(FLAGS_load);
	autoencoder_session->initialize();
	autoencoder_session->set_execution_context(execution_context);

	std::shared_ptr<Session> mnist_session = create_mnist_session();
	mnist_session->initialize();
	mnist_session->set_execution_context(execution_context);
	auto mnist_node = mnist_session->get_node("mnist_data");

	auto noise_session = create_noise_session();
	noise_session->initialize();
	execution_context->quit = false;
	noise_session->set_execution_context(execution_context);

	auto input = autoencoder_session->get_node("input");
	auto output = autoencoder_session->get_node("output");
	auto loss_node = autoencoder_session->get_node("loss");
	auto noise_node = noise_session->get_node("noise");


	int epoch;
	double avg_loss = 0;
	for (epoch = 1; epoch <= FLAGS_epoch && execution_context->quit != true; ++epoch) {
		
		execution_context->current_epoch = epoch;
		mnist_session->forward();
		input->feed_forward(mnist_node, 0);		
		autoencoder_session->forward();

		if (FLAGS_train) {
			avg_loss += loss_node->output(0)->value()->toFloat();
			autoencoder_session->backward(true, true);
			autoencoder_session->apply_solvers();
		}

		if (FLAGS_save != 0 && epoch % FLAGS_save == 0) {
			autoencoder_session->save("mnist_ac" + std::to_string(epoch) + ".bin");			
		}
	}

	if (FLAGS_save != 0) {
		autoencoder_session->save("mnist_ac" + std::to_string(epoch) + ".bin");
	}

	std::cout << "Average Loss: " << (avg_loss / epoch) << std::endl;
	
}

