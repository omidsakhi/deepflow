
#include "core/deep_flow.h"
#include "core/session.h"
#include "utilities/moving_average.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(size, 128, "Image size");
DEFINE_int32(z_dim, 1024, "Z Dimension");
DEFINE_int32(total, 200000, "Image channels"); // 13230
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 10000, "Maximum iterations");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save_image, 0, "Save image iter frequency (Don't Save = 0)");
DEFINE_int32(save_model, 0, "Save model iter frequency (Don't Save = 0)");
DEFINE_int32(print, 0, "Print source C++");

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
	df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, true, "face_data");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), "", "face_labels");	
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_z() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), "", "z");	
	return df.session();
}

std::shared_ptr<Session> create_static_z() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, -1, 1), "", "static_z");
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
	std::string err;	
	err = df.square_error(discriminator_input, labels_input);	
	auto loss = df.loss(err, DeepFlow::AVG);
	df.print({ loss }, "", DeepFlow::END_OF_EPOCH);
	return df.session();
}

std::string deconv(DeepFlow *df, std::string input, std::string a, std::string solver, int input_channels, int output_channels, int kernel, int pad, bool upsample, bool has_batchnorm, bool activation, std::string name) {

	auto node = input;
	if (upsample)
		node = df->resize(node, 2, 2, name + "_resize");
	auto f = df->variable(df->random_normal({ output_channels, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	node = df->conv2d(node, f, pad, pad, 1, 1, 1, 1, name + "_conv");
	if (has_batchnorm) {
		node = df->bias_add(node, output_channels, solver, name + "_bias");
		//node = df->batch_normalization(node, solver, output_channels, name + "_bn");
	}
	if (activation) {
		node = df->dprelu(node, a, name + "_relu");
		//return df->prelu(node, output_channels, solver, name + "_prelu");			
		//node = df->leaky_relu(node, 0.0, name + "_relu");
	}
	return node;
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, int stride, bool activation, std::string name) {
	stride = 1;
	auto f = df->variable(df->random_normal({ output_channels / 4, input_channels, kernel, kernel }, 0, 0.02), solver, name + "_f");
	auto node = df->conv2d(input, f, pad, pad, stride, stride, 1, 1, name + "_conv");
	node = df->patching(node, DeepFlow::PATCHING_DOWNCHANNELS, 2, 2, name + "_dp");
	node = df->bias_add(node, output_channels, solver, name + "_bias");
	if (activation) {
		//return df->prelu(node, output_channels, solver, name + "_prelu");
		return df->leaky_relu(node, 0.2, name + "_relu");		
	}
	return node;
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	int gf = 64;

	auto solver = df.adam_solver(0.00005f, 0.0f, 0.5f);
	
	auto node = df.place_holder({ FLAGS_batch, FLAGS_z_dim, 1, 1 }, Tensor::Float, "input");
	auto a = node;

	node = df.dense_with_bias(node, { FLAGS_z_dim, gf * 8 , 4 , 4 }, solver, "g41");	

	node = deconv(&df, node, a, solver, gf * 8, gf * 8, 3, 1, true, true, true, "g81"); // 8x8	

	node = deconv(&df, node, a, solver, gf * 8, gf * 4, 5, 2, true, true,true, "g161"); // 16x16

	node = deconv(&df, node, a, solver, gf * 4, gf * 2, 5, 2, true, true, true, "g321"); // 32x32	

	node = deconv(&df, node, a, solver, gf * 2, gf, 5, 2, true, true, true, "g641"); // 64x64	

	node = deconv(&df, node, a, solver, gf, FLAGS_channels, 5, 2, true, false, false, "g1281"); // 128x128	

	return df.session();

}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	 
	int dfs = 64;
	
	auto solver = df.adam_solver(0.00005f, 0.5f, 0.98f);

	auto node = df.place_holder({ FLAGS_batch , FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");		

	node = conv(&df, node, solver, 3, dfs * 2, 3, 1, 2, true, "d1");

	node = conv(&df, node, solver, dfs * 2, dfs * 4, 3, 1, 2, true, "d2");

	node = conv(&df, node, solver, dfs * 4, dfs * 8, 3, 1, 2,true, "d3");

	node = conv(&df, node, solver, dfs * 8, dfs * 16, 3, 1, 2, false, "d4");
	
	node = conv(&df, node, solver, dfs * 16, dfs * 16, 3, 1, 2, false, "d5");

	auto std = df.batch_stddev(node, "std");
	auto std_ones = df.variable(df.ones({ 1, FLAGS_batch, 4, 4 }), "", "d61");
	auto std_ip = df.matmul(std, std_ones, "d62");
	auto std_reshaped = df.reshape(std_ip, { FLAGS_batch, 1, 4, 4 }, "d63");	
	node = df.concate(node, std_reshaped, "d7");

	node = conv(&df, node, solver, dfs * 16 + 1, dfs * 16, 3, 1, 2, false, "d8");

	node = df.dense(node, { (dfs * 16) * 2 * 2, 1, 1, 1 }, solver, "d9");		

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

	if (FLAGS_print) {
		std::cout << generator->to_cpp() << std::endl;
		std::cout << discriminator->to_cpp() << std::endl;
		return;
	}

	auto imwrite_session = create_imwrite_session();
	imwrite_session->initialize(execution_context);

	auto face_reader = create_face_reader();
	face_reader->initialize(execution_context);

	auto face_labels = create_face_labels();
	face_labels->initialize(execution_context);

	auto generator_labels = create_generator_labels();
	generator_labels->initialize(execution_context);

	auto l2_loss_session = create_l2_loss();
	l2_loss_session->initialize(execution_context);	

	auto z_session = create_z();
	z_session->initialize(execution_context);

	auto static_z_session = create_static_z();
	static_z_session->initialize(execution_context);

	auto face_reader_data = face_reader->get_node("face_data");
	auto generator_output = generator->end_node(); 
	auto generator_input = generator->get_placeholder("input");
	auto discriminator_input = discriminator->get_placeholder("input");
	auto discriminator_output = discriminator->end_node();
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_node = face_labels->get_node("face_labels");
	auto l2_loss_discriminator_input = l2_loss_session->get_placeholder("discriminator_input");
	auto l2_loss_labels_input = l2_loss_session->get_placeholder("labels_input");
	auto l2_loss = std::dynamic_pointer_cast<Loss>(l2_loss_session->get_node("loss"));	
	auto imwrite_input = imwrite_session->get_placeholder("input");
	auto z = z_session->get_node("z");
	auto static_z = static_z_session->get_node("static_z");
	auto std = discriminator->get_node("std");

	int iter;
	if (FLAGS_load.empty()) {
		iter = 1;
	}
	else {
		iter = std::stoi(FLAGS_load) + 1;
	}	
	
	face_labels->forward();
	generator_labels->forward();	
	static_z->forward();
	
	MovingAverage g_loss_avg(100);
	MovingAverage d_loss_avg(100);
	
	float decay = 0.99999f;
	float lr = 0.00005f;

	for ( ; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {		

		lr *= decay;

		generator->set_learning_rate(lr);
		discriminator->set_learning_rate(lr);

		execution_context->current_iteration = iter;
		std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

		l2_loss->set_coef(1.0f);

		face_reader_data->forward();
		discriminator_input->write_values(face_reader_data->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss_session->forward();
		float d_loss = l2_loss->output(0)->value()->toFloat();
		float d_std = std->output(0)->value()->toFloat();
		l2_loss_session->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
		l2_loss_session->forward();
		d_loss += l2_loss->output(0)->value()->toFloat();
		d_loss_avg.add(d_loss);
		float g_std = std->output(0)->value()->toFloat();
		l2_loss_session->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();

		std::cout << " - d_loss: " << d_loss_avg.result();

		discriminator->apply_solvers();

		l2_loss->set_coef(1.0f);		

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss_session->forward();
		auto g_loss = l2_loss->output(0)->value()->toFloat();		
		l2_loss_session->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		
		l2_loss->set_coef(-1.0f);

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input->write_values(discriminator_output->output(0)->value());
		l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
		l2_loss_session->forward();
		g_loss += l2_loss->output(0)->value()->toFloat();
		l2_loss_session->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();

		g_loss_avg.add(g_loss);
		//std::cout << " - g_loss: " << g_loss_avg.result() << " | " << lr << std::endl;
		std::cout << " - g_loss: " << g_loss_avg.result() << " | " << lr << " | " << d_std << " | " << g_std << std::endl;		

		generator->apply_solvers();
		discriminator->reset_gradients();		

		if (FLAGS_save_image != 0 && iter % FLAGS_save_image == 0) {
			generator_input->write_values(static_z->output(0)->value());
			generator->forward();			
			imwrite_input->write_values(generator_output->output(0)->value());
			imwrite_session->forward();
		}

		if (FLAGS_save_model != 0 && iter % FLAGS_save_model == 0) {
			discriminator->save("d" + std::to_string(iter) + ".bin");
			generator->save("g" + std::to_string(iter) + ".bin");
		}

	}

	cudaDeviceReset();

}

