
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba", "Path to face 64x64 dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(total, 200000, "Image channels"); // 13230
DEFINE_int32(batch, 16, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 1000000, "Maximum iterations");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save, 0, "Save model iter frequency (Don't Save = 0)");

#define HAS_BIAS 1
#define NO_BIAS 0
#define HAS_BN 1
#define NO_BN 0
#define HAS_ACT 1
#define NO_ACT 0

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

std::shared_ptr<Session> create_face_reader() {
	DeepFlow df;
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, 64, 64 }, false, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_display_session(std::string name) {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, 3, 64, 64 }, Tensor::Float, "input");
	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, name);
	return df.session();
}

std::shared_ptr<Session> create_img_session() {
	DeepFlow df;
	auto solver = df.adam_solver(0.002f, 0.5f, 0.99f);
	auto gin = df.data_generator(df.random_uniform({ FLAGS_batch, 3, 64, 64 }, -1, 1), FLAGS_total, solver, "input");
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

std::string dense(DeepFlow *df, std::string input, std::initializer_list<int> dims, std::string solver, bool has_bias, bool has_bn, std::string name) {
	auto w = df->variable(df->random_normal(dims, 0, 0.02), solver, name + "_w");
	auto node = df->matmul(input, w);
	if (has_bias)
		node = bias(df, node, solver, *(dims.begin() + 1), name + "_bias");
	if (has_bn)
		node = batchnorm(df, node, solver, *(dims.begin() + 1), name + "_bn");
	return node;
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

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool has_bias, bool has_bn, bool activation, std::string name) {
	auto f = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	auto node = df->conv2d(input, f, pad, pad, stride, stride, 1, 1, name + "_conv");
	if (has_bn)
		node = batchnorm(df, node, solver, output_channels, name + "_bn");
	if (has_bias)
		node = bias(df, node, solver, output_channels, "_bias");
	if (activation) {
		return df->leaky_relu(node, 0.2, name + "_relu");
	}
	return node;
}

std::shared_ptr<Session> create_autoenc() {
	DeepFlow df;
	int flt = 64;
	auto solver = df.adam_solver(0.00002f, 0.5f, 0.99f);	

	auto node = df.place_holder({ FLAGS_batch , 3, 64, 64 }, Tensor::Float, "input");
	node = conv(&df, node, solver, 3, flt, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "e1");
	node = conv(&df, node, solver, flt, flt * 2, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "e2");
	node = conv(&df, node, solver, flt * 2, flt * 4, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "e3");
	node = conv(&df, node, solver, flt * 4, flt * 8, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "e4");
	node = dense(&df, node, { 512 * 4 * 4, 100, 1, 1 }, solver, HAS_BIAS, NO_BN, "e5");
	node = df.tanh(node, "z");
	node = dense(&df, node, { 100, flt * 8 , 4 , 4 }, solver, NO_BIAS, HAS_BN, "d1");
	node = deconv(&df, node, solver, flt * 8, flt * 4, 5, 2, 1, NO_BIAS, HAS_BN, HAS_ACT, "d2");
	node = deconv(&df, node, solver, flt * 4, flt * 2, 5, 2, 1, NO_BIAS, HAS_BN, HAS_ACT, "d3");
	node = deconv(&df, node, solver, flt * 2, flt, 5, 2, 1, NO_BIAS, HAS_BN, HAS_ACT, "d4");
	node = deconv(&df, node, solver, flt, 3, 5, 2, 1, NO_BIAS, HAS_BN, NO_ACT, "d5");
	node = df.tanh(node, "output");
	return df.session();
}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	int dfs = 64;
	
	auto solver = df.adam_solver(0.00002f, 0.5f, 0.99f);
	auto node = df.place_holder({ FLAGS_batch , 3, 64, 64 }, Tensor::Float, "input");
	node = conv(&df, node, solver, 3, dfs, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "di1");
	node = conv(&df, node, solver, dfs, dfs * 2, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "di2");
	node = conv(&df, node, solver, dfs * 2, dfs * 4, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "di3");
	node = conv(&df, node, solver, dfs * 4, dfs * 8, 5, 2, 2, HAS_BIAS, NO_BN, HAS_ACT, "di4");
	node = dense(&df, node, { 512 * 4 * 4, 1, 1, 1 }, solver, HAS_BIAS, NO_BN, "di5");	

	node = df.tanh(node, "di6");
	
	auto labels = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "label");
	auto coef = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "coef");
	auto err = df.reduce_mean(df.square_error(node, labels));
	//auto err = df.reduce_mean(df.abs(df.subtract(discriminator_input, labels_input)));
	auto loss = df.loss(err, coef, DeepFlow::AVG);	
	df.print({ loss }, " DISC LOSS {0}\n", DeepFlow::EVERY_PASS);

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> disc = create_discriminator();	
	disc->initialize(execution_context);

	std::shared_ptr<Session> ac = create_autoenc();
	ac->initialize(execution_context);
		
	auto img_session = create_img_session();
	img_session->initialize(execution_context);

	auto face_reader = create_face_reader();
	face_reader->initialize(execution_context);

	auto face_labels = create_face_labels();
	face_labels->initialize(execution_context);

	auto generator_labels = create_generator_labels();
	generator_labels->initialize(execution_context);

	auto display_session_r = create_display_session("real");
	display_session_r->initialize(execution_context);

	auto display_session_f = create_display_session("fake");
	display_session_f->initialize(execution_context);

	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_node = face_labels->get_node("face_labels");
	auto face_reader_data = face_reader->get_node("face_data");
	
	auto disc_input_img = disc->get_placeholder("input");
	auto disc_input_label = disc->get_placeholder("label");
	auto disc_coef = disc->get_placeholder("coef");

	auto ac_input_img = ac->get_placeholder("input");
	auto ac_output_img = ac->get_node("output");

	auto input_img = img_session->get_node("input");
	
	auto disp_input_r = display_session_r->get_node("input");
	auto disp_input_f = display_session_f->get_node("input");

	int iter = 1;	

	face_labels->forward();
	generator_labels->forward();
	disc_coef->fill(1.0);

	for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {
		std::cout << "Iteration: " << iter << std::endl;				

		img_session->forward();
		face_reader_data->forward();
		disc_input_img->write_values(face_reader_data->output(0)->value());
		disc_input_label->write_values(face_labels_node->output(0)->value());
		disc->forward();
		disc->backward();

		ac_input_img->write_values(input_img->output(0)->value());
		ac->forward();

		disc_input_img->write_values(ac_output_img->output(0)->value());
		disc_input_label->write_values(generator_labels_output->output(0)->value());
		disc->forward();
		disc->backward();
		disc->apply_solvers();

		disc_input_label->write_values(face_labels_node->output(0)->value());
		disc->forward();
		disc->backward();
		ac_output_img->write_diffs(disc_input_img->output(0)->diff());

		disp_input_r->write_values(disc_input_img->output(0)->diff());
		display_session_r->forward();

		disc->reset_gradients();
		ac->backward();
		input_img->write_diffs(ac_input_img->output(0)->diff());
		img_session->backward();
		ac->apply_solvers();

		img_session->apply_solvers();
		

		/*
		if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
			encoder->save("enc" + std::to_string(iter) + ".bin");
			generator->save("dec" + std::to_string(iter) + ".bin");
		}
		*/
	}

	/*
	if (FLAGS_save != 0) {
		generator->save("dec" + std::to_string(iter) + ".bin");
		encoder->save("enc" + std::to_string(iter) + ".bin");
	}
	*/


}

