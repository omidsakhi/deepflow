
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/face643", "Path to face 64x64 dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(total, 13230, "Image channels");
DEFINE_int32(batch, 15, "Batch size");
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

std::shared_ptr<Session> create_loss() {
	DeepFlow df;
	auto input1 = df.place_holder({ FLAGS_batch, 3, 64, 64 }, Tensor::Float, "input1");
	auto input2 = df.place_holder({ FLAGS_batch, 3, 64, 64 }, Tensor::Float, "input2");
	auto coef = df.place_holder({ 1, 1, 1, 1 }, Tensor::Float, "coef");	
	//auto meanabs = df.reduce_mean(df.square_error(input1, input2), "meanabs");
	auto meanabs = df.reduce_mean(df.abs(df.subtract(input1, input2)), "meanabs");
	auto loss = df.loss(meanabs, coef, DeepFlow::AVG, "output");
	df.print({ loss }, " LOSS {0}\n", DeepFlow::EVERY_PASS);
	return df.session();
}

std::shared_ptr<Session> create_display_session(std::string name) {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, 3, 64, 64 }, Tensor::Float, "input");
	df.display(input, 1, DeepFlow::EVERY_PASS, DeepFlow::VALUES, 1, name);
	return df.session();
}

std::string tconv(DeepFlow *df, std::string name, std::string input, std::string solver, int output_channels, int input_channels, int kernel, int pad, bool dropout, bool lift, bool activation, bool last) {
		
	auto var = df->variable(df->random_uniform({ output_channels * (lift?4:1), input_channels, kernel, kernel }, -0.1, 0.1), solver, name + "_f");
	std::string node = df->conv2d(input, var, pad, pad, 1, 1, 1, 1, name + "_t");
	auto bias = df->variable(df->step({ 1, output_channels * (lift ? 4 : 1), 1, 1 }, -0.1, 0.1), solver, name + "_b");
	node = df->bias_add(node, bias, name + "_b");
	if (lift)
		node = df->lifting(node, DeepFlow::LIFT_UP, name + "_l");
	if (dropout)
		node = df->dropout(node);
	if (activation) {		
		return df->elu(node, 0.01, last ? "output" : name + "_elu");
	}
	else {
		return df->leaky_relu(node, 1, last ? "output": name + "_relu");
	}
	return node;
}

std::string conv(DeepFlow *df, std::string name, std::string input, std::string solver, int output_channels, int input_channels, bool activation) {
	auto dconv_w = df->variable(df->random_uniform({ output_channels, input_channels, 3, 3 }, -0.1, 0.1), solver, name + "_w");
	auto dconv_t = df->conv2d(input, dconv_w, 1, 1, 2, 2, 1, 1, name);
	if (activation) {
		return df->elu(dconv_t, 0.01, name + "_elu");
	}
	return dconv_t;
}

std::shared_ptr<Session> create_z_session() {
	DeepFlow df;
	auto gin = df.data_generator(df.random_uniform({ FLAGS_batch, 100, 1, 1 }, -1, 1, "random_input"), FLAGS_total, "", "z");
	return df.session();
}

std::string t2conv(DeepFlow *df, std::string name, std::string input, std::string solver, int depth1, int depth2, int kernel, int pad, bool patching, bool activation) {
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
	//auto tconv_b = df->batch_normalization(tconv_t, DeepFlow::SPATIAL, 0.0f, 1.0f, 0.0f, 0.05f, 1.0f, name + "_b");
	if (activation) {
		auto drop = df->dropout(tconv_t, dropout);
		return df->relu(drop);
	}
	return tconv_t;
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;

	auto mean = 0;
	auto stddev = 0.01;
	auto dropout = 0.2;

	auto g_solver = df.adam_solver(0.00005f, 0.5f, 0.9f);
	auto input = df.place_holder({ FLAGS_batch , 100, 1, 1 }, Tensor::Float, "input");

	auto gfc_w = df.variable(df.random_uniform({ 100, 512, 4, 4 }, -stddev, stddev), g_solver, "gfc_w");
	auto gfc = df.matmul(input, gfc_w, "gfc");
	auto gfc_r = df.relu(gfc);
	auto gfc_drop = df.dropout(gfc_r, dropout);

	auto tconv1 = t2conv(&df, "tconv1", gfc_drop, g_solver, 512, 128, 2, 0, false, true);
	auto tconv2 = t2conv(&df, "tconv2", tconv1, g_solver, 256, 128, 3, 0, false, true);
	auto tconv3 = t2conv(&df, "tconv3", tconv2, g_solver, 256, 64, 5, 0, false, true);
	auto tconv4 = t2conv(&df, "tconv4", tconv3, g_solver, 128, 64, 6, 0, false, true);
	auto tconv5 = t2conv(&df, "tconv5", tconv4, g_solver, 3, 32, 7, 0, false, false);

	auto gout = df.tanh(tconv5, "output");	

	return df.session();
}

/*
std::shared_ptr<Session> create_generator() {
	DeepFlow df;		

	auto g_solver = df.adam_solver(0.0002f, 0.5f, 0.99f);	

	auto input = df.place_holder({ FLAGS_batch , 100, 1, 1 }, Tensor::Float, "input");

	auto gfc_w = df.variable(df.random_uniform({ 100, 128, 8, 8 }, -0.1, 0.1), g_solver, "gfc_w");
	auto gfc = df.matmul(input, gfc_w, "mul3");	

	auto conv31 = tconv(&df, "conv31", gfc, g_solver, 64, 128, 3, 1, false, false, true, false);
	auto conv32 = tconv(&df, "conv32", conv31, g_solver, 64, 64, 3, 1, false, true, true, false);
	auto conv33 = tconv(&df, "conv33", conv32, g_solver, 64, 64, 3, 1, false, true, true, false);
	auto conv34 = tconv(&df, "conv34", conv33, g_solver, 64, 64, 3, 1, false, true, true, false);
	auto conv35 = tconv(&df, "conv35", conv34, g_solver, 3, 64, 3, 1, false, false, false, true);	

	return df.session();
}
*/
std::shared_ptr<Session> create_critic() {
	DeepFlow df;
		
	auto d_solver = df.adam_solver(0.00005f, 0.5f, 0.999f);

	auto input = df.place_holder({ FLAGS_batch , FLAGS_channels, 64, 64 }, Tensor::Float, "input");

	auto conv11 = conv(&df, "conv11", input, d_solver, 32, 3, true);	
	auto conv12 = conv(&df, "conv12", conv11, d_solver, 64, 32, true);
	auto conv13 = conv(&df, "conv13", conv12, d_solver, 128, 64, true);
	auto conv14 = conv(&df, "conv14", conv13, d_solver, 256, 128, true);

	auto fc1_w = df.variable(df.random_uniform({ 4096, 100, 1, 1 }, -0.1, 0.1), d_solver, "fc1_w");		
	auto fc1 = df.matmul(conv14, fc1_w, "mul1");

	auto fc2_w = df.variable(df.random_uniform({ 100, 256, 8, 8 }, -0.1, 0.1), d_solver, "fc2_w");		
	auto fc2 = df.matmul(fc1, fc2_w, "mul2");

	auto conv21 = tconv(&df, "conv21", fc2, d_solver, 64, 256, 3, 1, false, false, true, false);
	auto conv22 = tconv(&df, "conv22", conv21, d_solver, 64, 64, 3, 1, true, true, true, false);
	auto conv23 = tconv(&df, "conv23", conv22, d_solver, 64, 64, 3, 1, true, true, true, false);
	auto conv24 = tconv(&df, "conv24", conv23, d_solver, 64, 64, 3, 1, true, true, true, false);
	auto conv25 = tconv(&df, "conv25", conv24, d_solver, 3, 64, 3, 1, false, false, false, false);	

	df.tanh(conv25, "output");

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> critic;
	if (FLAGS_load.empty())
		critic = create_critic();
	else
		critic = load_generator_session(FLAGS_load);
	critic->initialize(execution_context);

	std::shared_ptr<Session> generator;
	if (FLAGS_load.empty())
		generator = create_generator();
	else
		generator = load_disc_session(FLAGS_load);
	generator->initialize(execution_context);

	auto z_session = create_z_session();
	z_session->initialize(execution_context);

	auto face_reader = create_face_reader();
	face_reader->initialize(execution_context);

	auto loss = create_loss();
	loss->initialize(execution_context);

	auto display_session_r = create_display_session("real");
	display_session_r->initialize(execution_context);

	auto display_session_f = create_display_session("fake");
	display_session_f->initialize(execution_context);

	auto face_reader_data = face_reader->get_node("face_data");
	auto z_output = z_session->get_node("z");
	
	auto crit_input_img = critic->get_placeholder("input");
	auto crit_output_img = critic->get_node("output");
	
	auto gen_input_z = generator->get_placeholder("input");
	auto gen_output_img = generator->get_node("output");
	
	auto loss_input1 = loss->get_node("input1");	
	auto loss_input2 = loss->get_node("input2");
	auto loss_output = loss->get_node("output");
	auto loss_coef = loss->get_node("coef");

	auto disp_input_r = display_session_r->get_node("input");
	auto disp_input_f = display_session_f->get_node("input");


	int iter = 1;
	float kt = 0;

	z_session->forward();

	for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {
		std::cout << "Iteration: " << iter << std::endl;				

		face_reader_data->forward();
		crit_input_img->write_values(face_reader_data->output(0)->value());
		critic->forward();
		disp_input_r->write_values(crit_output_img->output(0)->value());		
		display_session_r->forward();		
		loss_input1->write_values(crit_output_img->output(0)->value());
		loss_input2->write_values(face_reader_data->output(0)->value());
		loss_coef->fill(1.0f);
		loss->forward();
		float real_loss = loss_output->output(0)->value()->toFloat();
		loss->backward();
		crit_output_img->write_diffs(loss_input1->output(0)->diff());
		critic->backward();		

		std::cout << " Kt: " << kt << std::endl;
		z_session->forward();
		gen_input_z->write_values(z_output->output(0)->value());
		generator->forward();
		disp_input_f->write_values(gen_output_img->output(0)->value());
		display_session_f->forward();
		crit_input_img->write_values(gen_output_img->output(0)->value());
		critic->forward();
		loss_input1->write_values(crit_output_img->output(0)->value());
		loss_input2->write_values(gen_output_img->output(0)->value());
		loss_coef->fill(-1.0 * kt);
		loss->forward();
		float fake_loss = loss_output->output(0)->value()->toFloat();
		loss->backward();
		crit_output_img->write_diffs(loss_input1->output(0)->diff());
		critic->backward();
		critic->apply_solvers();		
		loss_coef->fill(1.0f);
		loss->backward();
		crit_output_img->write_diffs(loss_input1->output(0)->diff());
		critic->backward();
		gen_output_img->write_diffs(crit_input_img->output(0)->diff());
		generator->backward();
		generator->apply_solvers();
		critic->reset_gradients();		

		kt = kt + 0.001 * (0.3 * real_loss - fake_loss);
		if (kt < 0)
			kt = 0;
		else if (kt > 1)
			kt = 1;

		if (FLAGS_save != 0 && iter % FLAGS_save == 0) {
			generator->save("d" + std::to_string(iter) + ".bin");
			critic->save("g" + std::to_string(iter) + ".bin");
		}
	}

	if (FLAGS_save != 0) {
		generator->save("d" + std::to_string(iter) + ".bin");
		critic->save("g" + std::to_string(iter) + ".bin");
	}


}

