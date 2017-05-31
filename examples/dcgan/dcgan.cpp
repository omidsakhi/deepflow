
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(model, "D:/Projects/deepflow/build/x64/Release/models/VGG_ILSVRC_16_layers.caffemodel", "Path to VGG16 model");
DEFINE_string(image_folder, "D:/Projects/deepflow/data/face", "Path to image folder dataset");
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
DEFINE_int32(epoch, 1000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");


void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		

	if (FLAGS_i.empty()) {
		auto train = df.define_train_phase("Train");
		auto g_solver = df.adadelta_solver();
		auto d_solver = df.sgd_solver(0.98f, 0.00001f);

		auto mean = 0;
		auto stddev = 0.02;
		auto negative_slope = 0.2;

		auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 10, 7, 7 }, mean, stddev, "random_input"), 100, "", "input", { train });

		auto gconv1_f = df.variable(df.random_normal({ 10, 1024, 3, 3 }, mean, stddev), g_solver, "gconv1_f", { train });
		auto gconv1 = df.transposed_conv2d(gin, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1", { train });
		auto gconv_relu = df.leaky_relu(gconv1);		
		
		auto gconv2_f = df.variable(df.random_normal({ 1024, 1, 3, 3 }, mean, stddev), g_solver, "gconv2_f", { train });
		auto gconv2 = df.transposed_conv2d(gconv_relu, gconv2_f, 1, 1, 2, 2, 1, 1, "gconv2", { train });
		
		auto gout = df.tanh(gconv2);
		
		auto neg = df.negate(gout, DeepFlow::DIFFS);

		auto din = df.mnist_reader("D:/Projects/deepflow/data/mnist", 100, MNISTReader::MNISTReaderType::Test, MNISTReader::Data);
		
		auto select = df.data_generator(df.random_uniform({ 1,1,1,1 }, 0.5f, 1.99), 10000, "", "select");
		auto data_selector = df.multiplexer({ neg, din }, select, "data_selector");
				
		auto disp = df.display(neg);

		auto dconv1_f = df.variable(df.random_normal({ 32, 1, 5, 5 }, mean, stddev), d_solver, "dconv1_f");
		auto dconv1 = df.conv2d(data_selector, dconv1_f, 0, 0, 1, 1, 1, 1, "dconv1");		
		auto dconv1_relu = df.leaky_relu(dconv1, negative_slope, "dconv1_relu");
		auto dconv2_f = df.variable(df.random_normal({ 64, 32, 5, 5 }, mean, stddev), d_solver, "dconv2_f");
		auto dconv2 = df.conv2d(dconv1_relu, dconv2_f, 0, 0, 1, 1, 1, 1, "dconv2");
		auto dconv2_relu = df.leaky_relu(dconv2, negative_slope, "dconv2_relu");
		auto dconv3_f = df.variable(df.random_normal({ 128, 64, 5, 5 }, mean, stddev), d_solver, "dconv3_f");
		auto dconv3 = df.conv2d(dconv2_relu, dconv3_f, 0, 0, 1, 1, 1, 1, "dconv3");
		auto dfc1_w = df.variable(df.random_normal({ 32768, 128, 1, 1 }, mean, stddev), d_solver, "dfc1_w");
		auto dfc1_m = df.matmul(dconv3, dfc1_w, "dfc1_m");
		auto dfc1_b = df.variable(df.random_normal({ 1, 128, 1, 1 }, mean, stddev), d_solver, "dfc1_b");
		auto dfc1_bias = df.bias_add(dfc1_m, dfc1_b, "dfc1_bias");
		auto dfc2_w = df.variable(df.random_normal({ 128, 1, 1, 1 }, mean, stddev), d_solver, "dfc2_w");
		auto dfc2 = df.matmul(dfc1_bias, dfc2_w, "dfc2_m");
		
		auto d_labels = df.data_generator( df.random_uniform({ batch_size, 1, 1, 1}, 0.5, 1.0), 10000, "", "true_labels");
		auto g_labels = df.data_generator(df.random_uniform({ batch_size, 1, 1, 1 }, -1.0, -0.5), 10000, "", "fake_labels");

		auto label_selector = df.multiplexer({ g_labels, d_labels }, select, "label_selector");
		
		df.euclidean_loss(dfc2, label_selector);
		
	}
	else {
		df.block()->load_from_binary(FLAGS_i);	
	}
	
	auto session = df.session();	

	if (!FLAGS_run.empty()) {
		session->initialize();
		session->run(FLAGS_run, FLAGS_epoch, FLAGS_iter, FLAGS_printiter, FLAGS_printepoch, FLAGS_debug);
	}

	if (FLAGS_cpp) {
		session->initialize();
		std::cout << session->to_cpp() << std::endl;
	}

	
	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.block()->save_as_text(FLAGS_o);
		else
			df.block()->save_as_binary(FLAGS_o);
	}
	
}

