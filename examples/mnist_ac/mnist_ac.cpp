
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

std::shared_ptr<Session> load_session(std::string suffix) {
	std::string filename = "mnist_ac" + suffix + ".bin";
	std::cout << "Loading generator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> create_autoencoder_session() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.02;
	auto enc_solver = df.adam_solver(0.0001f, 0.5f);
	auto dec_solver = df.adam_solver(0.0001f, 0.5f);
	auto enc_negative_slope = 0.01;
	auto dec_negative_slope = 0.01;		
	auto alpha_param = 1.0f;
	auto decay = 0.001f;

	auto input = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");	 

	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input");

	auto conv1_w = df.variable(df.random_normal({ 16, 1, 3, 3 }, mean, stddev), enc_solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 1, 1, 1, 1, 1, 1, "conv1");	
	auto conv1_r = df.leaky_relu(conv1, enc_negative_slope);
	auto conv1_p = df.pooling(conv1_r, 2, 2, 0, 0, 2, 2, "conv1_p");
	
	auto conv2_w = df.variable(df.random_normal({ 8, 16, 3, 3 }, mean, stddev), enc_solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_p, conv2_w, 1, 1, 1, 1, 1, 1, "conv2");	
	auto conv2_r = df.leaky_relu(conv2, enc_negative_slope);
	auto conv2_p = df.pooling(conv2_r, 2, 2, 0, 0, 2, 2, "conv2_p");

	auto conv3_w = df.variable(df.random_normal({ 8, 8, 3, 3 }, mean, stddev), enc_solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_p, conv3_w, 1, 1, 1, 1, 1, 1, "conv3");
	auto conv3_r = df.leaky_relu(conv3, enc_negative_slope);
	auto conv3_p = df.pooling(conv3_r, 2, 2, 1, 1, 2, 2, "conv3_p");

	auto tconv1_f = df.variable(df.random_normal({ 8, 8, 2, 2 }, mean, stddev), dec_solver, "tconv1_f");
	auto tconv1_t = df.transposed_conv2d(conv3_p, tconv1_f, 1, 1, 2, 2, 1, 1, "tconv1_t");
	auto tconv1_r = df.leaky_relu(tconv1_t, dec_negative_slope);

	auto tconv2_f = df.variable(df.random_normal({ 8, 16, 3, 3 }, mean, stddev), dec_solver, "tconv2_f");
	auto tconv2_t = df.transposed_conv2d(tconv1_r, tconv2_f, 1, 1, 2, 2, 1, 1, "tconv2_t");
	auto tconv2_r = df.leaky_relu(tconv2_t, dec_negative_slope);

	auto tconv3_f = df.variable(df.random_normal({ 16, 1, 3, 3 }, mean, stddev), dec_solver, "tconv3_f");
	auto tconv3_t = df.transposed_conv2d(tconv2_r, tconv3_f, 1, 1, 2, 2, 1, 1, "tconv3_t");
	auto tconv3_r = df.leaky_relu(tconv3_t, dec_negative_slope);
	
	auto output = df.tanh(tconv3_r);
	df.display(output, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "output");
	auto euc = df.euclidean_distance(output, input);
	auto loss = df.loss(euc, DeepFlow::AVG);
	
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

	int epoch;
	for (epoch = 1; epoch <= FLAGS_epoch && execution_context->quit != true; ++epoch) {
		
		execution_context->current_epoch = epoch;		
		
		autoencoder_session->reset_gradients();
		autoencoder_session->forward();
		autoencoder_session->backward();
		autoencoder_session->apply_solvers();

		if (FLAGS_save != 0 && epoch % FLAGS_save == 0) {
			autoencoder_session->save("mnist_ac" + std::to_string(epoch) + ".bin");			
		}
	}

	if (FLAGS_save != 0) {
		autoencoder_session->save("mnist_ac" + std::to_string(epoch) + ".bin");
	}

}

