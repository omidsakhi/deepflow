
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "D:/Projects/deepflow/data/mnist", "Path to mnist folder dataset");
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
		
		auto solver_on = df.data_generator(df.fill({ 1,1,1,1 }, 2.0), 10000, "", "solver_on");
		auto solver_off = df.data_generator(df.fill({ 1,1,1,1 }, 0.0), 10000, "", "solver_off");

		auto select = df.data_generator(df.random_uniform({ 1,1,1,1 }, 0.6, 1.99), 10000, "", "select");
		
		auto d_solver_enable = df.multiplexer({ solver_off, solver_on }, select, "d_solver_enable");
		auto g_solver_enable = df.multiplexer({ solver_on, solver_off }, select, "d_solver_enable");

		auto d_solver = df.gain_solver(0.9999f, 0.00100, 10.000000, 0.000000, 0.050000, 0.950000, "d_solver", d_solver_enable);
		auto g_solver = df.adam_solver(0.001f, 0.5f, 0.98f, 1.0e-7, "g_solver", g_solver_enable);

		auto mean = 0;
		auto stddev = 0.002;
		auto negative_slope = 0.1;

		auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, mean, stddev, "random_input"), 10000, "", "input", { train });				

		auto gfc_w = df.variable(df.random_normal({ 100, 1024, 4, 4 }, mean, stddev), g_solver, "gfc_w");		
		auto gfc = df.matmul(gin, gfc_w, "gfc");				

		auto gconv1_f = df.variable(df.random_normal({ 1024, 512, 2, 2 }, mean, stddev), g_solver, "gconv1_f", { train });		
		auto gconv1 = df.transposed_conv2d(gfc, gconv1_f, 1, 1, 2, 2, 1, 1, "gconv1", { train });					
		auto gconv1_relu = df.leaky_relu(gconv1);
		auto gconv2_f = df.variable(df.random_normal({ 512, 128, 3, 3 }, mean, stddev), g_solver, "gconv2_f", { train });		
		auto gconv2 = df.transposed_conv2d(gconv1_relu, gconv2_f, 1, 1, 2, 2, 1, 1, "gconv2", { train });
		auto gconv2_relu = df.leaky_relu(gconv2);
		auto gconv3_f = df.variable(df.random_normal({ 128, 1, 3, 3 }, mean, stddev), g_solver, "gconv3_f", { train });		
		auto gconv3 = df.transposed_conv2d(gconv2_relu, gconv3_f, 1, 1, 2, 2, 1, 1, "gconv3", { train });
		
		auto gout = df.tanh(gconv3);				

		auto din = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::MNISTReaderType::Test, MNISTReader::Data);				
		
		auto data_selector = df.multiplexer({ gout, din }, select, "data_selector");
				
		auto disp_selector = df.display(data_selector, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, "disp_select");
		
		auto conv1_w = df.variable(df.random_uniform({ 16, 1, 3, 3 }, -0.100000, 0.100000), d_solver, "conv1_w", {});
		auto conv1 = df.conv2d(data_selector, conv1_w, 1, 1, 2, 2, 1, 1, "conv1", {});
		auto conv1_relu = df.leaky_relu(conv1, negative_slope);
		auto conv2_w = df.variable(df.random_uniform({ 32, 16, 3, 3 }, -0.100000, 0.100000), d_solver, "conv2_w", {});
		auto conv2 = df.conv2d(conv1_relu, conv2_w, 1, 1, 2, 2, 1, 1, "conv2", {});
		auto conv2_relu = df.leaky_relu(conv2, negative_slope);
		auto conv3_w = df.variable(df.random_uniform({ 64, 32, 3, 3 }, -0.100000, 0.100000), d_solver, "conv3_w", {});
		auto conv3 = df.conv2d(conv2_relu, conv3_w, 1, 1, 2, 2, 1, 1, "conv3", {});
		auto conv3_relu = df.leaky_relu(conv3, negative_slope); 
		auto w1 = df.variable(df.random_uniform({ 1024, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1", {});
		auto m1 = df.matmul(conv3_relu, w1, "m1", {});
		auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1", {});
		auto bias1 = df.bias_add(m1, b1, "bias1", {});
		auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1", {});
		auto dropout = df.dropout(relu1, 0.500000, true, "dropout", {});
		auto w2 = df.variable(df.random_uniform({ 500, 10, 1, 1 }, -0.100000, 0.100000), d_solver, "w2", {});
		auto m2 = df.matmul(dropout, w2, "m2", {});
		auto b2 = df.variable(df.step({ 1, 10, 1, 1 }, -1.000000, 1.000000), d_solver, "b2", {});
		auto bias2 = df.bias_add(m2, b2, "bias2", {});
		auto relu2 = df.leaky_relu(bias2, negative_slope, "relu2", {});

		auto data_labels = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Test, MNISTReader::Labels, "test_labels");		
		auto fake_labels = df.data_generator(df.random_uniform({ batch_size, 10, 1, 1 }, 0.7, 1.1), 10000, "", "fake_labels");
		
		auto label_selector = df.multiplexer({ fake_labels, data_labels }, select, "label_selector");

		auto loss = df.softmax_loss(relu2, label_selector, "loss", {});
		auto predict = df.argmax(loss, 1, "predict", {});
		auto target = df.argmax(label_selector, 1, "target", {});
		auto equal = df.equal(predict, target, "equal", {});
		auto acc = df.accumulator(equal, DeepFlow::END_OF_EPOCH, "acc", {});
		auto correct = df.reduce_sum(acc[0], 0, "correct", {});
		df.print({ correct, acc[1] }, "    TRAINSET   CORRECT COUNT : {0} OUT OF {1} = %{}%\n", DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, "print");		
		
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

