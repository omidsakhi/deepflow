
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(size, 128, "Image size");
DEFINE_int32(total, 2000000, "Image channels"); // 13230
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 10000, "Maximum iterations");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save_image, 0, "Save image iter frequency (Don't Save = 0)");
DEFINE_int32(save_model, 0, "Save model iter frequency (Don't Save = 0)");

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
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, false, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), FLAGS_total, "", "face_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), FLAGS_total, "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_imwrite_session() {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");
	df.imwrite(input, "{it}", "imwrite");
	return df.session();
}

std::shared_ptr<Session> create_l2_loss() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto labels_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "labels_input");
	auto coef = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "coef");
	std::string err;
	err = df.reduce_mean(df.square_error(discriminator_input, labels_input));
	auto loss = df.loss(err, coef, DeepFlow::AVG);
	df.print({ loss }, "", DeepFlow::END_OF_EPOCH);
	return df.session();
}

std::string batchnorm(DeepFlow *df, std::string input, std::string solver, int output_channels, std::string name) {
	auto bns = df->variable(df->random_normal({ 1, output_channels, 1, 1 }, 1, 0.02), solver, name + "_bns");
	auto bnb = df->variable(df->fill({ 1, output_channels, 1, 1 }, 0), solver, name + "_bnb");
	return df->batch_normalization(input, bns, bnb, DeepFlow::SPATIAL, true, name + "_bn");
}

std::string dense(DeepFlow *df, std::string input, std::initializer_list<int> dims, std::string solver, std::string name) {
	auto w = df->variable(df->random_normal( dims, 0, 0.02), solver, name + "_w");
	return df->matmul(input, w);
}

std::string deconv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, bool upsample, bool activation, std::string name) {

	auto node = input;
	if (upsample)
		node = df->resize(node, 2, 2, name + "_resize");
	auto f = df->variable(df->random_normal({ output_channels, upsample? input_channels : input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	node = df->conv2d(node, f, pad, pad, 1, 1, 1, 1, name + "_conv");
	node = batchnorm(df, node, solver, output_channels, name + "_bn");
	if (activation) {
		return df->elu(node, 0.1, name + "_elu");
	}
	return node;
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool activation, std::string name) {
	auto f = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	auto node = df->conv2d(input, f, pad, pad, stride, stride, 1, 1, name + "_conv");	
	auto bnb = df->variable(df->fill({ 1, output_channels, 1, 1 }, 0), solver, name + "_bnb");
	node = df->bias_add(node, bnb, name + "_bias");
	if (activation) {
		return df->leaky_relu(node, 0.2, name + "_relu");		
	}
	return node;
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	int gf = 64;

	auto solver = df.adam_solver(0.0002f, 0.5f, 0.9f);

	auto node = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1), FLAGS_total, "", "input");

	node = dense(&df, node, { 100, gf * 16 , 4 , 4 }, solver, "g1");
	node = batchnorm(&df, node, solver, gf * 16, "g2"); // 4x4

	node = deconv(&df, node, solver, gf * 16, gf * 8, 3, 1, 1, 1, "g3"); // 8x8	

	node = deconv(&df, node, solver, gf * 8, gf * 4, 5, 2, 1, 1, "g4"); // 16x16	

	node = deconv(&df, node, solver, gf * 4, gf * 2, 5, 2, 1, 1, "g5"); // 32x32	

	node = deconv(&df, node, solver, gf * 2, gf * 2, 5, 2, 1, 1, "g7"); // 64x64	

	node = deconv(&df, node, solver, gf * 2, FLAGS_channels, 5, 2, 1, 0, "g9"); // 128x128

	node = df.tanh(node, "gout");

	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;

	int dfs = 64;

	auto solver = df.adam_solver(0.0002f, 0.5f, 0.9f);

	auto node = df.place_holder({ FLAGS_batch , FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");

	node = conv(&df, node, solver, 3, dfs, 5, 2, 2, true, "d1");

	node = conv(&df, node, solver, dfs, dfs * 2, 5, 2, 2, true, "d2");

	node = conv(&df, node, solver, dfs * 2, dfs * 4, 5, 2, 2, true, "d3");

	node = conv(&df, node, solver, dfs * 4, dfs * 8, 5, 2, 2, true, "d4");

	node = dense(&df, node, { (dfs * 8) * 8 * 8, 1, 1, 1 }, solver, "d5");

	node = df.sigmoid(node, "dout");

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

	auto imwrite_session = create_imwrite_session();
	imwrite_session->initialize(execution_context);

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
	auto imwrite_input = imwrite_session->get_placeholder("input");

	int iter = 1;

	loss_coef->fill(1.0f);
	face_labels->forward();
	generator_labels->forward();

	for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {

		execution_context->current_iteration = iter;
		std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

		for (int k = 0; k < 2; k++) {
			face_reader_data->forward();
			discriminator_input->write_values(face_reader_data->output(0)->value());
			discriminator->forward();
			l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
			l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
			l2_loss->forward();
			float d_loss = l2_loss_output->output(0)->value()->toFloat();
			l2_loss->backward();
			discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
			discriminator->backward();

			generator->forward();
			discriminator_input->write_values(generator_output->output(0)->value());
			discriminator->forward();
			l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
			l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
			l2_loss->forward();
			d_loss += l2_loss_output->output(0)->value()->toFloat();
			l2_loss->backward();
			discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
			discriminator->backward();

			std::cout << " - d_loss: " << d_loss;

		}
		discriminator->apply_solvers();

		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss->forward();
		auto g_loss = l2_loss_output->output(0)->value()->toFloat();
		std::cout << " - g_loss: " << g_loss << std::endl;
		l2_loss->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		generator->apply_solvers();
		discriminator->reset_gradients();

		if (FLAGS_save_image != 0 && iter % FLAGS_save_image == 0) {
			imwrite_input->write_values(generator_output->output(0)->value());
			imwrite_session->forward();
		}

		if (FLAGS_save_model != 0 && iter % FLAGS_save_model == 0) {
			discriminator->save("d" + std::to_string(iter) + ".bin");
			generator->save("g" + std::to_string(iter) + ".bin");
		}

	}


}

