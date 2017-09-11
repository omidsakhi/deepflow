
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
	auto sqerr = df.square_error(loss_t1, loss_t2);
	auto loss = df.loss(sqerr, DeepFlow::AVG);

	df.print({ loss }, "Epoch: {ep} - Loss: {0}\n", DeepFlow::EVERY_PASS);

	return df.session();
}

std::shared_ptr<Session> create_display_session() {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "input");
	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, "disp");
	return df.session();
}

std::string conv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, bool activation) {
	auto mean = 0;
	auto stddev = 0.001;
	auto conv_w = df->variable(df->random_normal({ depth1, depth2, 3, 3 }, mean, stddev), solver, name + "_w");
	auto conv_ = df->conv2d(input, conv_w, 1, 1, 1, 1, 1, 1, name);
	auto conv_p = df->pooling(conv_, 2, 2, 0, 0, 2, 2, name + "_p");
	if (activation)
		return  df->elu(conv_p, 0.01);
	return conv_p;
}

std::string tconv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, int kernel, int pad, bool activation) {

	auto mean = 0;
	auto stddev = 0.001;
	auto tconv_l = df->lifting(input, DeepFlow::LIFT_UP, name + "_l");
	auto tconv_f = df->variable(df->random_normal({ depth1, depth2, kernel, kernel }, mean, stddev), solver, name + "_f");
	auto tconv_t = df->conv2d(tconv_l, tconv_f, pad, pad, 1, 1, 1, 1, name + "_t");
	auto tconv_b = df->batch_normalization(tconv_t, DeepFlow::SPATIAL, 0.2, 1, 0, 0, 1);
	if (activation)
		return  df->relu(tconv_b);
	return tconv_b;

}

std::shared_ptr<Session> create_encoder_session() {
	DeepFlow df;
	auto solver = df.adam_solver(0.00005f, 0.9f);

	auto input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "enc_input");
	auto conv1 = conv(&df, "conv1", input, solver, 256, 1, true);
	auto conv2 = conv(&df, "conv2", conv1, solver, 512, 256, true);
	auto conv3 = conv(&df, "conv3", conv2, solver, 1024, 512, true);
	
	auto drop = df.dropout(conv3, 0.1f);
	
	auto fc_w = df.variable(df.random_normal({ 1024 * 3 * 3, FLAGS_bottleneck, 1, 1 }, 0, 0.1), solver, "fc_w");
	auto fc = df.matmul(drop, fc_w, "fc");

	auto enc_out = df.tanh(fc, "enc_output");

	return df.session();
}

std::shared_ptr<Session> create_decoder_session() {

	DeepFlow df;
	auto solver = df.adam_solver(0.00005f, 0.9f);

	auto dec_input = df.place_holder({ FLAGS_batch, FLAGS_bottleneck, 1, 1 }, Tensor::Float, "dec_input");

	auto fc_w = df.variable(df.random_normal({ FLAGS_bottleneck, 1024, 3, 3 }, 0, 0.01), solver, "fc_w");
	auto fc = df.matmul(dec_input, fc_w, "fc");
	
	auto drop = df.dropout(fc, 0.1f);

	auto tconv1 = tconv(&df, "tconv1", drop, solver, 512, 256, 2, 0, true);
	auto tconv2 = tconv(&df, "tconv2", tconv1, solver, 512, 128, 2, 0, true);
	auto tconv3 = tconv(&df, "tconv3", tconv2, solver, 256, 128, 3, 0, true);
	auto tconv4 = tconv(&df, "tconv4", tconv3, solver, 1, 64, 5, 0, false);

	auto output = df.tanh(tconv4, "dec_output");	

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;	

	std::shared_ptr<Session> encoder_session;
	encoder_session = create_encoder_session();
	encoder_session->initialize(execution_context);	

	std::shared_ptr<Session> decoder_session;
	decoder_session = create_decoder_session();
	decoder_session->initialize(execution_context);	

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
	auto enc_input = encoder_session->get_node("enc_input");
	auto enc_output = encoder_session->get_node("enc_output");
	auto dec_input = decoder_session->get_node("dec_input");
	auto dec_output = decoder_session->get_node("dec_output");
	auto disp_input = display_session->get_node("input");

	auto loss_t1 = loss_session->get_node("loss_t1");
	auto loss_t2 = loss_session->get_node("loss_t2");

	int epoch;
	for (epoch = 1; epoch <= FLAGS_epoch && execution_context->quit != true; ++epoch) {
		
		execution_context->current_epoch = epoch;
	
		mnist_train_session->forward();
		enc_input->write_values(mnist_train_data->output(0)->value());
		encoder_session->forward();
		dec_input->write_values(enc_output->output(0)->value());
		decoder_session->forward();

		//disp_input->write_values(dec_output->output(0)->value());
		//display_session->forward();

		loss_session->reset_gradients();
		loss_t1->write_values(dec_output->output(0)->value());
		loss_t2->write_values(mnist_train_data->output(0)->value());
		loss_session->forward();
		loss_session->backward();
		decoder_session->reset_gradients();
		dec_output->write_diffs(loss_t1->output(0)->diff());
		decoder_session->backward();
		decoder_session->apply_solvers();
		encoder_session->reset_gradients();
		enc_output->write_diffs(dec_input->output(0)->diff());
		encoder_session->backward();
		encoder_session->apply_solvers();

		mnist_test_session->forward();
		enc_input->write_values(mnist_test_data->output(0)->value());
		encoder_session->forward();
		dec_input->write_values(enc_output->output(0)->value());
		decoder_session->forward();
		disp_input->write_values(dec_output->output(0)->value());
		display_session->forward();
	}

}
