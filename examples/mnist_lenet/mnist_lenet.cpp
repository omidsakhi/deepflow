
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(mnist, "C:/Projects/deepflow/data/mnist", "Path to MNIST data folder");
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

std::string conv(DeepFlow *df, std::string name, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, bool activation) {
	auto w = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.1), solver, name + "_w");
	auto node = df->conv2d(input, w, pad, pad, 1, 1, 1, 1, name);
	//auto bns = df->variable(df->random_uniform({ 1, output_channels, 1, 1 }, -0.1, 0.1), solver, name + "_bns");
	auto bnb = df->variable(df->fill({ 1, output_channels, 1, 1 }, 0), solver, name + "_bnb");
	node = df->bias_add(node, bnb);
	//node = df->batch_normalization(node, bns, bnb, DeepFlow::SPATIAL, false, name + "_b");
	node = df->pooling(node, 2, 2, 0, 0, 2, 2);
	if (activation) {
		node = df->elu(node, 0.1);
		//node = df->dropout(node, 0.5f);
		return node; 
	}
	return node;
}


void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		
	if (FLAGS_i.empty()) {		

		df.define_phase("Train", DeepFlow::TRAIN);
		df.define_phase("Validation", DeepFlow::VALIDATION);

		auto solver = df.adam_solver(0.00001f, 0.9f);
		
		auto test_data = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Test, MNISTReader::Data, "test_data", { "Validation" });		
		auto train_data = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Train, MNISTReader::Data, "train_data", { "Train" });
		auto data_selector = df.phaseplexer(train_data, "Train", test_data, "Validation", "data_selector", {});
				
		auto conv1 = conv(&df, "conv1", data_selector, solver, 1, 32, 5,2, true);		
		auto conv2 = conv(&df, "conv2", conv1, solver, 32, 64, 5,2, true);

		auto w1 = df.variable(df.random_uniform({ 7 * 7 * 64, 1024, 1, 1 }, -0.1, 0.1), solver, "w1", {});		
		auto b1 = df.variable(df.fill({ 1, 1024, 1, 1 }, 0), solver, "b1");
		auto m1 = df.bias_add(df.matmul(conv2, w1, "m1", {}), b1);

		auto r1 = df.relu(m1);
		auto d1 = df.dropout(r1);

		auto w2 = df.variable(df.random_uniform({ 1024, 10, 1, 1 }, -0.1, 0.1), solver, "w2");
		auto m2 = df.matmul(d1, w2, "m2");

		auto softmax = df.softmax(m2);
		auto softmax_log = df.log(softmax, -1.0f);
		auto train_labels = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Train, MNISTReader::Labels, "train_labels", { "Train" });
		auto test_labels = df.mnist_reader(FLAGS_mnist, 100, MNISTReader::Test, MNISTReader::Labels, "test_labels", { "Validation" });
		auto label_selector = df.phaseplexer(train_labels, "Train", test_labels, "Validation", "label_selector", {});		
		auto softmax_dot = df.dot(softmax_log, label_selector);
		auto loss = df.loss(softmax_dot, DeepFlow::AVG, "loss");
		auto predict = df.argmax(softmax, 1, "predict", {});
		auto target = df.argmax(label_selector, 1, "target", {});
		auto equal = df.equal(predict, target, "equal", {});
		auto acc = df.accumulator(equal, DeepFlow::END_OF_EPOCH, "acc", {});
		auto correct = df.reduce_sum(acc[0], 0, "correct", {});
		df.print({ correct, acc[1] }, "    TRAINSET   CORRECT COUNT : {0} OUT OF {1} = %{}%\n", DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, "print1", { "Train" });
		df.print({ correct, acc[1] }, "    VALIDATION CORRECT COUNT : {0} OUT OF {1} = %{}%\n", DeepFlow::END_OF_EPOCH, DeepFlow::VALUES, "print2", { "Validation" });
		
	}
	else {
		df.block()->load_from_binary(FLAGS_i);	
	}
	
	df.block()->print_node_params();
	df.block()->print_phase_params();

	auto session = df.session();	

	if (!FLAGS_run.empty()) {
		session->initialize();
		//session->run(FLAGS_run, FLAGS_epoch, FLAGS_iter, FLAGS_printiter, FLAGS_printepoch, FLAGS_debug);
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

