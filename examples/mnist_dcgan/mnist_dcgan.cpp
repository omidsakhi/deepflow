
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

std::shared_ptr<Session> load_generator_session(std::string suffix) {
	std::string filename = "g" + suffix + ".bin";
	std::cout << "Loading generator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> load_disc_session(std::string suffix) {
	std::string filename = "d" + suffix + ".bin";
	std::cout << "Loading discriminator from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::shared_ptr<Session> create_mnist_reader() {
	DeepFlow df;
	df.mnist_reader(FLAGS_mnist, FLAGS_batch, MNISTReader::MNISTReaderType::Train, MNISTReader::Data, "mnist_data");
	return df.session();
}

std::shared_ptr<Session> create_disc_pos_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.9, 1.0), 60000, "", "labels");	
	return df.session();
}

std::shared_ptr<Session> create_disc_neg_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.1), 60000, "", "labels");
	return df.session();
}

std::shared_ptr<Session> create_gen_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.05), 60000, "", "labels");
	return df.session();
}

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto euc = df.square_error(discriminator_input, labels_input);
	auto loss = df.loss(euc, DeepFlow::SUM, "loss");
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
		return  df->leaky_relu(conv_p, 0.1);
	return conv_p;
}

std::string tconv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, int kernel, int pad, bool activation) {
	auto mean = 0;
	auto stddev = 0.001;
	auto tconv_l = df->lifting(input, DeepFlow::LIFT_UP, name + "_l");
	auto tconv_f = df->variable(df->random_normal({ depth1, depth2, kernel, kernel }, mean, stddev), solver, name + "_f");	
	auto tconv_t = df->conv2d(tconv_l, tconv_f, pad, pad, 1, 1, 1, 1, name + "_t");
	auto tconv_b = df->batch_normalization(tconv_t, DeepFlow::SPATIAL, 0.2, 1, 0, 0, 1, name + "_b");	
	if (activation)
		return df->relu(tconv_b);
	return tconv_b;
}

std::shared_ptr<Session> create_generator_session() {
	DeepFlow df;
	auto solver = df.adam_solver(0.00002f, 0.5f);

	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1, "random_input"), 60000, "", "input");

	auto fc_w = df.variable(df.random_normal({ 100, 512, 3, 3 }, 0, 0.001), solver, "fc_w");
	auto fc = df.matmul(gin, fc_w, "fc");

	auto tconv1 = tconv(&df, "tconv1", fc, solver, 256, 128, 2, 0, true);
	auto tconv2 = tconv(&df, "tconv2", tconv1, solver, 256, 64, 2, 0, true);
	auto tconv3 = tconv(&df, "tconv3", tconv2, solver, 256, 64, 3, 0, true);
	auto tconv4 = tconv(&df, "tconv4", tconv3, solver, 1, 64, 5, 0, false);

	auto output = df.tanh(tconv4, "gout");

	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;	
	auto solver = df.adam_solver(0.00002f, 0.5f);

	auto input = df.place_holder({ FLAGS_batch, 1, 28, 28 }, Tensor::Float, "input");
	auto conv1 = conv(&df, "conv1", input, solver, 128, 1, true);
	auto conv2 = conv(&df, "conv2", conv1, solver, 256, 128, true);
	auto conv3 = conv(&df, "conv3", conv2, solver, 1024, 256, true);
	
	auto fc_w = df.variable(df.random_normal({ 1024 * 3 * 3, 1, 1, 1 }, 0, 0.001), solver, "fc_w");
	auto fc = df.matmul(conv3, fc_w, "fc");

	auto enc_out = df.sigmoid(fc, "dout");

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	
	
	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> generator;
	if (FLAGS_load.empty())
		generator = create_generator_session();
	else
		generator = load_generator_session(FLAGS_load);			
	generator->initialize(execution_context);	

	std::shared_ptr<Session> discriminator;

	if (FLAGS_load.empty())
		discriminator = create_discriminator();
	else
		discriminator = load_disc_session(FLAGS_load);		
	discriminator->initialize(execution_context);	

	auto mnist_reader = create_mnist_reader();
	mnist_reader->initialize(execution_context);	

	auto disc_pos_labels = create_disc_pos_labels();
	disc_pos_labels->initialize(execution_context);

	auto disc_neg_labels = create_disc_neg_labels();
	disc_neg_labels->initialize(execution_context);

	auto gen_labels = create_gen_labels();
	gen_labels->initialize(execution_context);

	auto loss = create_loss();
	loss->initialize(execution_context);	

	auto socket_io_sender_session = create_socket_io_sender();
	socket_io_sender_session->initialize(execution_context);

	auto disp_session = create_display_session();
	disp_session->initialize(execution_context);
	

	auto mnist_reader_data = mnist_reader->get_node("mnist_data");
	auto generator_output = generator->get_node("gout");	
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->get_node("dout");
	auto disc_neg_labels_output = disc_neg_labels->get_node("labels");
	auto disc_pos_labels_output = disc_pos_labels->get_node("labels");
	auto gen_labels_output = gen_labels->get_node("labels");	
	auto loss_discriminator_input = loss->get_placeholder("discriminator_input");
	auto loss_labels_input = loss->get_placeholder("labels_input");
	auto loss_output = loss->get_node("loss");
	auto sio_sender_dtrue = socket_io_sender_session->get_node("dtrue");
	auto sio_sender_dfake = socket_io_sender_session->get_node("dfake");
	auto sio_sender_gfake = socket_io_sender_session->get_node("gfake");
	auto disp = disp_session->get_node("input");

	int iter;
	for (iter = 1; iter <= FLAGS_epoch && execution_context->quit != true; ++iter) {
		
		execution_context->current_epoch = iter;

		std::cout << "Epoch: " << iter << std::endl;		
		
		std::cout << " DISC POS " << std::endl;
		mnist_reader_data->forward();
		discriminator_input->write_values(mnist_reader_data->output(0)->value());
		discriminator->forward();
		disc_pos_labels_output->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(disc_pos_labels_output->output(0)->value());
		loss->forward();
		sio_sender_dtrue->write_values(loss_output->output(0)->value());
		loss->backward();		
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers();		

		std::cout << " DISC NEG " << std::endl;
		generator->forward();
		disp->write_values(generator_output->output(0)->value());
		disp_session->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		disc_neg_labels_output->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(disc_neg_labels_output->output(0)->value());
		loss->forward();
		sio_sender_dfake->write_values(loss_output->output(0)->value());
		loss->backward();		
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers(); 		

		std::cout << " GEN " << std::endl;
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		gen_labels_output->forward();
		loss->reset_gradients();
		loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		loss_labels_input->write_values(gen_labels_output->output(0)->value());
		loss->forward();
		sio_sender_gfake->write_values(loss_output->output(0)->value());
		loss->backward();
		discriminator->reset_gradients();
		discriminator_output->write_diffs(loss_discriminator_input->output(0)->diff());
		discriminator->backward();	
		generator->reset_gradients();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();		

		socket_io_sender_session->forward();

		if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
			discriminator->save("d" + std::to_string(iter) + ".bin");
			generator->save("g" + std::to_string(iter) + ".bin");
		}
	}

	if (FLAGS_save != 0) {
		discriminator->save("d" + std::to_string(iter) + ".bin");
		generator->save("g" + std::to_string(iter) + ".bin");
	}

}

