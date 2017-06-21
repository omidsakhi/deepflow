
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
	auto solver = df.adam_solver(0.0001f, 0.9f, 0.999f);
	auto negative_slope = 0.1;
	auto dropout = 0.5; 
	auto depth = 64; 
	auto alpha_param = 1.0f;
	auto decay = 0.0001f;

	auto input = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");	 

	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "input");

	auto conv1_w = df.variable(df.random_uniform({ depth, 1, 5, 5 }, -0.100000, 0.100000), solver, "conv1_w");
	auto conv1 = df.conv2d(input, conv1_w, 2, 2, 2, 2, 1, 1, "conv1");	
	auto conv1_r = df.leaky_relu(conv1, negative_slope);	
	
	auto conv2_w = df.variable(df.random_uniform({ depth*2, depth, 3, 3 }, -0.100000, 0.100000), solver, "conv2_w");
	auto conv2 = df.conv2d(conv1_r, conv2_w, 1, 1, 2, 2, 1, 1, "conv2");	
	auto conv2_r = df.leaky_relu(conv2, negative_slope);	

	auto conv3_w = df.variable(df.random_uniform({ depth*2, depth*2, 3, 3 }, -0.100000, 0.100000), solver, "conv3_w");
	auto conv3 = df.conv2d(conv2_r, conv3_w, 1, 1, 2, 2, 1, 1, "conv3");
	auto conv3_r = df.leaky_relu(conv3, negative_slope);	
	auto conv3_d = df.dropout(conv3_r, dropout);

	auto enc_w1 = df.variable(df.random_uniform({ depth*2*4*4, 500, 1, 1 }, -0.100000, 0.100000), solver, "enc_w1");
	auto enc_m1 = df.matmul(conv3_d, enc_w1, "enc_m1");
	auto enc_b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), solver, "enc_b1");
	auto enc_bias1 = df.bias_add(enc_m1, enc_b1, "enc_bias1");
	auto enc_relu1 = df.leaky_relu(enc_bias1, negative_slope);
	auto enc_drop1 = df.dropout(enc_relu1, dropout);

	auto enc_w2 = df.variable(df.random_uniform({ 500, 100, 1, 1 }, -0.100000, 0.100000), solver, "w2");
	auto enc_m2 = df.matmul(enc_drop1, enc_w2, "m2");
	auto enc_b2 = df.variable(df.step({ 1, 100, 1, 1 }, -1.000000, 1.000000), solver, "b2");
	auto enc_bias2 = df.bias_add(enc_m2, enc_b2, "latent");	

	auto gfc_w = df.variable(df.random_normal({ 100, depth, 4, 4 }, mean, stddev), solver, "gfc_w");
	auto gfc = df.matmul(enc_bias2, gfc_w, "gfc");
	auto gfc_r = df.leaky_relu(gfc, negative_slope, true);	

	auto gconv1_f = df.variable(df.random_normal({ depth, depth / 2, 2, 2 }, mean, stddev), solver, "gconv1_f");
	auto gconv1_t = df.transposed_conv2d(gfc_r, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1");
	auto gconv1_n = df.batch_normalization(gconv1_t, DeepFlow::SPATIAL, 0, 1, 0, alpha_param, 1 - decay);
	auto gconv1_r = df.leaky_relu(gconv1_n, negative_slope, true);	

	auto gconv2_f = df.variable(df.random_normal({ depth / 2, depth / 4, 3, 3 }, mean, stddev), solver, "gconv2_f");
	auto gconv2_t = df.transposed_conv2d(gconv1_r, gconv2_f, 1, 1, 2, 2, 1, 1, "gconv2");
	auto gconv2_n = df.batch_normalization(gconv2_t, DeepFlow::SPATIAL, 0, 1, 0, alpha_param, 1 - decay);
	auto gconv2_r = df.leaky_relu(gconv2_n, negative_slope, true);	

	auto gconv3_f = df.variable(df.random_normal({ depth / 4, 1, 3, 3 }, mean, stddev), solver, "gconv3_f");
	auto gconv3_t = df.transposed_conv2d(gconv2_r, gconv3_f, 1, 1, 2, 2, 1, 1, "gconv3");
	auto gconv3_n = df.batch_normalization(gconv3_t, DeepFlow::PER_ACTIVATION, 0, 1, 0, alpha_param, 1 - decay);

	auto gout = df.tanh(gconv3_n, "gout");

	df.display(gout, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "output");

	auto euc = df.euclidean_distance(gout, input);
	auto loss = df.loss(euc, DeepFlow::SUM, "loss");
	df.print({ loss }, " NORM {0}\n", DeepFlow::EVERY_PASS);

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

	int iter;
	for (iter = 1; iter <= FLAGS_epoch && execution_context->quit != true; ++iter) {
		
		execution_context->current_epoch = iter;

		std::cout << "Epoch: " << iter << std::endl;		
		
		autoencoder_session->reset_gradients();
		autoencoder_session->forward();
		autoencoder_session->backward();
		autoencoder_session->apply_solvers();

		if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
			autoencoder_session->save("mnist_ac" + std::to_string(iter) + ".bin");			
		}
	}

	if (FLAGS_save != 0) {
		autoencoder_session->save("mnist_ac" + std::to_string(iter) + ".bin");
	}

}

