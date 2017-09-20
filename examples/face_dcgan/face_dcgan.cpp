
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/face643", "Path to face 64x64 dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(total, 13230, "Image channels");
DEFINE_int32(batch, 49, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 1000000, "Maximum iterations");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save, 0, "Save model iter frequency (Don't Save = 0)");

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

std::shared_ptr<Session> create_face_reader() {
	DeepFlow df;
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, 64, 64 }, true, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0.8, 1.0), FLAGS_total, "", "face_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 1, 1, 1 }, 0, 0.2), FLAGS_total, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_l2_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto coef = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "coef");
	auto sqerr = df.square_error(discriminator_input, labels_input);
	auto loss = df.loss(sqerr, coef, DeepFlow::AVG);	
	df.print({ loss }, " NORM {0}\n", DeepFlow::EVERY_PASS); 
	return df.session();
}

std::string tconv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, int kernel, int pad, bool patching, bool activation) {
	auto mean = 0;
	auto stddev = 0.01;
	auto dropout = 0.5;	
	std::string tconv_l;
	if (patching)
		tconv_l = df->patching(input, DeepFlow::PATCHING_UP, 2, 2, name + "_l");
	else
		tconv_l = df->lifting(input, DeepFlow::LIFT_UP, name + "_l");
	auto tconv_f = df->variable(df->random_uniform({ depth1, depth2, kernel, kernel }, -stddev, stddev), solver, name + "_f");
	auto tconv_t = df->conv2d(tconv_l, tconv_f, pad, pad, 1, 1, 1, 1, name + "_t");
	auto tconv_b = df->batch_normalization(tconv_t, DeepFlow::SPATIAL, 0.0f, 1.0f, 0.0f, -0.05f, 1.0f, name + "_b");
	if (activation) {
		auto drop = df->dropout(tconv_b, dropout);
		return df->relu(drop);
	}
	return tconv_b;
}

std::string conv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, bool activation) {
	auto mean = 0;
	auto stddev = 0.01;
	auto negative_slope = 0.1;
	auto dropout = 0.5;
	auto conv_w = df->variable(df->random_uniform({ depth1, depth2, 3, 3 }, -stddev, stddev), solver, name + "_w");
	auto conv_ = df->conv2d(input, conv_w, 1, 1, 2, 2, 1, 1, name);
	if (activation) {
		auto relu = df->elu(conv_, negative_slope);
		return df->dropout(relu, dropout);
	}
	return conv_;
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;

	auto mean = 0;
	auto stddev = 0.01;
	auto dropout = 0.2;

	auto g_solver = df.adam_solver(0.0002f, 0.0f, 0.99f);
	auto gin = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1, "random_input"), FLAGS_total, "", "input");

	auto gfc_w = df.variable(df.random_uniform({ 100, 512, 4, 4 }, -stddev, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(gin, gfc_w, "gfc");
	auto gfc_r = df.relu(gfc);
	auto gfc_drop = df.dropout(gfc_r, dropout);

	auto tconv1 = tconv(&df, "tconv1", gfc_drop, g_solver, 512, 128, 2, 0,false, true);
	auto tconv2 = tconv(&df, "tconv2", tconv1, g_solver, 256, 128, 3, 0,false, true);
	auto tconv3 = tconv(&df, "tconv3", tconv2, g_solver, 256, 64, 5, 0,false, true);
	auto tconv4 = tconv(&df, "tconv4", tconv3, g_solver, 128, 64, 6, 0,false, true);
	auto tconv5 = tconv(&df, "tconv5", tconv4, g_solver, 3, 32, 7, 0,false, false);

	auto gout = df.tanh(tconv5, "gout");
	auto disp = df.display(gout, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES);

	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	auto mean = 0;
	auto stddev = 0.01;
	auto d_solver = df.adam_solver(0.0002f, 0.0f, 0.99f);
	auto negative_slope = 0.1;
	auto input = df.place_holder({ FLAGS_batch , FLAGS_channels, 64, 64 }, Tensor::Float, "input");

	auto conv1 = conv(&df, "conv1", input, d_solver, 64, 3, true);
	auto conv2 = conv(&df, "conv2", conv1, d_solver, 64, 64, true);
	auto conv3 = conv(&df, "conv3", conv2, d_solver, 128, 64, true);
	auto conv4 = conv(&df, "conv4", conv3, d_solver, 256, 128, true);	

	auto w1 = df.variable(df.random_uniform({ 4096, 500, 1, 1 }, -0.100000, 0.100000), d_solver, "w1");
	auto m1 = df.matmul(conv4, w1, "m1");
	auto b1 = df.variable(df.step({ 1, 500, 1, 1 }, -1.000000, 1.000000), d_solver, "b1");
	auto bias1 = df.bias_add(m1, b1, "bias1");
	auto relu1 = df.leaky_relu(bias1, negative_slope, "relu1");
	auto drop2 = df.dropout(relu1);

	auto w2 = df.variable(df.random_uniform({ 500, 1, 1, 1 }, -0.100000, 0.100000), d_solver, "w2");
	auto m2 = df.matmul(drop2, w2, "m2");

	auto dout = df.sigmoid(m2, "dout");

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> generator;
	if (FLAGS_load.empty())
		generator = create_generator();
	else
		generator = load_generator_session(FLAGS_load);
	generator->initialize(execution_context);

	std::shared_ptr<Session> discriminator;

	if (FLAGS_load.empty())
		discriminator = create_discriminator();
	else
		discriminator = load_disc_session(FLAGS_load);
	discriminator->initialize(execution_context);

	auto face_reader = create_face_reader();
	face_reader->initialize(execution_context);

	auto face_labels = create_face_labels();
	face_labels->initialize(execution_context);

	auto generator_labels = create_generator_labels();
	generator_labels->initialize(execution_context);

	auto l2_loss = create_l2_loss();
	l2_loss->initialize(execution_context);	

	auto face_reader_data = face_reader->get_node("face_data");
	auto generator_output = generator->get_node("gout");
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->get_node("dout");
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_node = face_labels->get_node("face_labels");
	auto l2_loss_discriminator_input = l2_loss->get_placeholder("discriminator_input");
	auto l2_loss_labels_input = l2_loss->get_placeholder("labels_input");
	auto l2_loss_output = l2_loss->get_node("loss");
	auto loss_coef = l2_loss->get_node("coef");
	loss_coef->fill(1.0f);

	int iter = 1;
	for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {
		std::cout << "Iteration: " << iter << std::endl;

		std::cout << " MNIST INPUT " << std::endl;
		face_reader_data->forward();
		discriminator_input->write_values(face_reader_data->output(0)->value());
		discriminator->forward();
		face_labels->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss->forward();		
		l2_loss->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();		

		std::cout << " GENERATOR INPUT " << std::endl;
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		generator_labels->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
		l2_loss->forward();		
		l2_loss->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		discriminator->apply_solvers();

		std::cout << " TRAINING GENERATOR " << std::endl;		
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		face_labels->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss->forward();		
		l2_loss->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();
		discriminator->reset_gradients();

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

