
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
DEFINE_int32(bottleneck, 16, "Bottleneck Channel Size");

std::shared_ptr<Session> load_session(std::string prefix, std::string suffix) {
	std::string filename = prefix + suffix + ".bin";
	std::cout << "Loading "<< prefix << " from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> create_mnist_train_session() {
	DeepFlow df;
	df.mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");
	return df.session();
}

std::shared_ptr<Session> create_mnist_test_session() {
	DeepFlow df;
	df.mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Test, MNISTReader::Data, "mnist_data");
	return df.session();
}

std::shared_ptr<Session> create_loss_session() {
	
	DeepFlow df;
	
	auto loss_t1 = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "loss_t1");
	auto loss_t2 = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "loss_t2");
	auto sqerr = df.reduce_mean(df.square_error(loss_t1, loss_t2));
	//auto sqerr = df.square_error(loss_t1, loss_t2);
	auto loss = df.loss(sqerr,"", DeepFlow::AVG);

	df.print({ loss }, "Epoch: {ep} - Loss: {0}\n", DeepFlow::EVERY_PASS);

	return df.session();
}

std::shared_ptr<Session> create_display_session() {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "input");
	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp");
	return df.session();
}
std::string bias(DeepFlow *df, std::string input, std::string solver, int output_channels, std::string name) {	
	return df->bias_add(input, df->variable(df->fill({ 1, output_channels, 1, 1 }, 0), solver, name + "_bw"));
}

std::string batchnorm(DeepFlow *df, std::string input, std::string solver, int output_channels, std::string name) {
	auto bns = df->variable(df->random_normal({ 1, output_channels, 1, 1 }, 1, 0.02), solver, name + "_bns");
	auto bnb = df->variable(df->fill({ 1, output_channels, 1, 1 }, 0), solver, name + "_bnb");
	return df->batch_normalization(input, bns, bnb, DeepFlow::SPATIAL, true, name + "_bn");
}

std::string dense(DeepFlow *df, std::string input, std::initializer_list<int> dims, std::string solver, std::string name) {
	auto w = df->variable(df->random_normal(dims, 0, 0.02), solver, name + "_w");
	return df->matmul(input, w);
}

std::string deconv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, bool upsample, bool has_bias, bool has_bn, bool activation, std::string name) {

	auto node = input;
	auto f = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	if (upsample)
		node = df->upsample(node);
	node = df->conv2d(node, f, pad, pad, 1, 1, 1, 1, name + "_conv");
	if (has_bn)
		node = batchnorm(df, node, solver, output_channels, name + "_bn");
	if (has_bias)
		node = bias(df, node, solver, output_channels, "_bias");
	if (activation) {
		return df->relu(node);
	}
	return node;
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool activation, std::string name) {
	auto f = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	auto node = df->conv2d(input, f, pad, pad, stride, stride, 1, 1, name + "_conv");
	node = batchnorm(df, node, solver, output_channels, name + "_bn");
	if (activation) {
		return df->leaky_relu(node, 0.2, name + "_relu");
	}
	return node;
}

std::shared_ptr<Session> create_session() {
	DeepFlow df;

	int flt = 64;

	auto solver = df.adam_solver(0.00002f, 0.7f, 0.99f);

	auto input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "enc_input");
	auto node = conv(&df, input, solver, 1, flt, 5, 2, 2, 1, "e1"); // 14 * 14 
	auto c1 = node;
	node = conv(&df, node, solver, flt, flt * 2, 5, 2, 2, 1, "e2"); // 7 * 7
	auto c2 = node;
	node = conv(&df, node, solver, flt * 2, flt * 4, 5, 2, 2, 1, "e2");
	auto c3 = node;
	node = dense(&df, node, { (flt*4) * 4 * 4, 100, 1, 1 }, solver, "e3");
	node = bias(&df, node, solver, 100, "e4");
	node = df.tanh(node, "h");
	node = dense(&df, node, { 100, flt * 4, 4, 4 }, solver, "d1");
	node = batchnorm(&df, node, solver, flt * 4, "d2");
	node = deconv(&df, node, solver, flt * 4, flt * 2, 2, 0, 1, 1, 0, 1, "d4");
	node = deconv(&df, node, solver, flt * 2, flt, 5, 2, 1, 1, 0, 1, "d6");
	node = deconv(&df, node, solver, flt, 1, 5, 2, 1, 1, 0, 0, "d8");

	auto output = df.tanh(node, "dec_output");	

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;	

	std::shared_ptr<Session> model;
	if (FLAGS_load.empty())
		model = create_session();
	else
		model = load_session("m", FLAGS_load);
	model->initialize(execution_context);

	std::shared_ptr<Session> mnist_train_session = create_mnist_train_session();
	mnist_train_session->initialize(execution_context);	

	std::shared_ptr<Session> mnist_test_session = create_mnist_test_session();
	mnist_test_session->initialize(execution_context);	

	std::shared_ptr<Session> loss_session = create_loss_session();
	loss_session->initialize(execution_context);	
	
	auto display_session = create_display_session();
	display_session->initialize(execution_context);	
	
	auto mnist_train_data = mnist_train_session->get_node("mnist_data");
	auto mnist_test_data = mnist_test_session->get_node("mnist_data");
	auto enc_input = model->get_node("enc_input");
	auto dec_output = model->get_node("dec_output");
	auto disp_input = display_session->get_node("input");

	auto loss_t1 = loss_session->get_node("loss_t1");
	auto loss_t2 = loss_session->get_node("loss_t2");
	
	int epoch;
	for (epoch = 1; epoch <= FLAGS_epoch && execution_context->quit != true; ++epoch) {
		
		execution_context->current_epoch = epoch;
	
		mnist_train_session->forward();
		enc_input->write_values(mnist_train_data->output(0)->value());
		model->forward();

		//disp_input->write_values(dec_output->output(0)->value());
		//display_session->forward();
		
		loss_t1->write_values(dec_output->output(0)->value());
		loss_t2->write_values(mnist_train_data->output(0)->value());
		loss_session->forward();
		loss_session->backward();		
		dec_output->write_diffs(loss_t1->output(0)->diff());
		loss_session->reset_gradients();
		model->backward();
		model->apply_solvers();
		
		mnist_test_session->forward();
		enc_input->write_values(mnist_test_data->output(0)->value());
		model->forward();
		disp_input->write_values(dec_output->output(0)->value());
		display_session->forward();
	
		if (FLAGS_save != 0 && epoch % FLAGS_save == 0) {
			model->save("ac" + std::to_string(epoch) + ".bin");			
		}

	}

	if (FLAGS_save != 0) {
		model->save("ac" + std::to_string(epoch) + ".bin");		
	}

}
