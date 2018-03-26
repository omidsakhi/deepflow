
#include "core/deep_flow.h"
#include "core/session.h"
#include "utilities/moving_average.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(size, 128, "Image size");
DEFINE_int32(total, 200000, "Total Images"); // 13230
DEFINE_int32(batch, 40, "Batch size");
DEFINE_int32(debug, 0, "Debug Level");
DEFINE_int32(iter, 200000, "Maximum iterations");
DEFINE_int32(sigmaiter, 100000, "Sigma iteration");
DEFINE_double(maxsigma, 3, "Maximum Sigma");
DEFINE_double(minsigma, 0.1, "Minimum Sigma");
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
	auto imbatch = df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, true, "imbatch");
	auto gaussian_kernel = df.gaussian_kernel(63, 10, "gaussian_kernel");
	df.conv2d(imbatch, gaussian_kernel, 31, 31, 1, 1, 1, 1, "face_data");	
	return df.session();
}

std::shared_ptr<Session> create_z() {
	DeepFlow df;
	df.data_generator(df.truncated_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1),"", "z");	
	return df.session();
}

std::shared_ptr<Session> create_static_z() {
	DeepFlow df;
	df.data_generator(df.truncated_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1),"", "static_z");
	return df.session();
}


std::shared_ptr<Session> create_imwrite_session() {
	DeepFlow df;
	auto input = df.place_holder({ FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");
	df.imwrite(input, "{it}", "imwrite");
	return df.session();
}

std::shared_ptr<Session> create_face_labels() {
	DeepFlow df;
	df.data_generator(df.truncated_normal({ FLAGS_batch, 1, 1, 1 }, 1, 1), "", "face_labels");
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.truncated_normal({ FLAGS_batch, 1, 1, 1 }, -1, 1), "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_l2_loss() {
	DeepFlow df;
	auto discriminator_mean = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_mean");
	auto discriminator_sigma = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_sigma");
	auto target_dist = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "target_dist");
	auto discriminator_dist = df.gaussian(discriminator_mean, discriminator_sigma, "discriminator_dist");
	std::string err;
	err = df.reduce_mean(df.square_error(discriminator_dist, target_dist));
	auto loss = df.loss(err, DeepFlow::AVG);
	df.print({ loss }, "", DeepFlow::END_OF_EPOCH);
	return df.session();
}

std::string deconv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, bool activation, std::string name) {
	auto pad = floor(kernel / 2);
	auto node = df->conv2d_with_bias(input, input_channels, output_channels, kernel, pad, 1, solver, name);
	if (activation) {		
		node = df->lrn(node, name + "_lrn", 1);
		node = df->prelu(node, output_channels, solver, name);
	}
	return node;
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int stride, bool activation, std::string name) {
	auto node = input;
	node = df->conv2d_with_bias(node, input_channels, output_channels, 3, 1, stride, solver, name);
	if (activation) {		
		node = df->prelu(node, output_channels, solver, name);
	}
	return node;
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	int gf = 64;

	auto solver = df.adam_solver(0.00005f, 0.0f, 0.98f);
	
	auto node = df.place_holder({ FLAGS_batch, 100, 1, 1 }, Tensor::Float, "input");

	node = df.dense_with_bias(node, { 100, gf * 16 , 4 , 4 }, solver, "g4"); 	
	node = deconv(&df, node, solver, gf * 16, gf * 8, 3, true, "g41");	
	node = df.resize(node, 2.0f, 2.0f, "g8");
	node = deconv(&df, node, solver, gf *  8, gf * 8, 3, true, "g81");	
	node = df.resize(node, 2.0f, 2.0f, "g16");
	node = deconv(&df, node, solver, gf * 8, gf * 4, 5, true, "g161");	
	node = df.resize(node, 2.0f, 2.0f, "g32");
	node = deconv(&df, node, solver, gf * 4, gf * 2, 5, true, "g321");	
	node = df.resize(node, 2.0f, 2.0f, "g64");
	node = deconv(&df, node, solver, gf * 2, gf, 5, true, "g641");	
	node = df.resize(node, 2.0f, 2.0f, "g128");	
	node = deconv(&df, node, solver, gf, FLAGS_channels, 5, false, "g1282");

	return df.session();

}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;
	 
	int dfs = 64;

	auto solver = df.adam_solver(0.00005f, 0.5f, 0.98f);

	auto node = df.place_holder({ FLAGS_batch , FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");			
	
	node = df.patch_sampling(node, 64, 64, "d0");

	node = conv(&df, node, solver, 3, dfs * 4, 2, true, "d1"); // 32x32

	node = conv(&df, node, solver, dfs * 4, dfs * 4, 2, true, "d2"); // 16x16

	node = conv(&df, node, solver, dfs * 4, dfs * 8, 2, true, "d3"); // 8x8

	node = conv(&df, node, solver, dfs * 8, dfs * 8, 2, true, "d4"); // 4x4	

	auto mean = df.dense(node, { (dfs * 8) * 4 * 4, 1, 1, 1 }, solver, "dmean");
	df.pass_through(mean, false, "output_mean");

	auto sigma = df.dense(node, { (dfs * 8) * 4 * 4, 1, 1, 1 }, solver, "dsigma");
	df.pass_through(sigma, false, "output_sigma");

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

	auto face_reader_session = create_face_reader();
	face_reader_session->initialize(execution_context);

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

	auto face_data = face_reader_session->get_node("face_data");
	auto face_gaussian_kernel = std::dynamic_pointer_cast<ConvGaussianKernel>(face_reader_session->get_node("gaussian_kernel"));
	auto generator_output = generator->get_node("g1282_conv");	
	auto generator_input = generator->get_placeholder("input");
	auto discriminator_input = discriminator->get_placeholder("input");	
	auto discriminator_output_mean = discriminator->get_node("output_mean");
	auto discriminator_output_sigma = discriminator->get_node("output_sigma");
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_node = face_labels->get_node("face_labels");
	auto l2_loss_discriminator_mean = l2_loss_session->get_placeholder("discriminator_mean");
	auto l2_loss_discriminator_sigma = l2_loss_session->get_placeholder("discriminator_sigma");
	auto l2_loss_labels_input = l2_loss_session->get_placeholder("target_dist");
	auto l2_loss = std::dynamic_pointer_cast<Loss>(l2_loss_session->get_node("loss"));	
	auto imwrite_input = imwrite_session->get_placeholder("input");
	auto z = z_session->get_node("z");	
	auto static_z = static_z_session->get_node("static_z");	

	int iter;
	if (FLAGS_load.empty()) {
		iter = 1;
	}
	else {
		iter = std::stoi(FLAGS_load) + 1;
	}	
	
	static_z->forward();
	
	MovingAverage g_loss_avg(100);
	MovingAverage p_d_loss_avg(100);
	MovingAverage n_d_loss_avg(100);
	
	float decay = 0.99999f;
	float lr = 0.0001f;	
	
	for ( ; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {		

		float sigma = FLAGS_maxsigma * exp( (float) iter / FLAGS_sigmaiter * log(FLAGS_minsigma / FLAGS_maxsigma) );
		sigma = floor(sigma * 100.0f) / 100.0f;
		if (sigma < FLAGS_minsigma)
			sigma = FLAGS_minsigma;
		face_gaussian_kernel->setSigma(sigma);

		lr *= decay;

		generator->set_learning_rate(lr);
		discriminator->set_learning_rate(lr);

		execution_context->current_iteration = iter;
		std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

		l2_loss->set_coef(1.0f);

		face_reader_session->forward();
		face_labels_node->forward();		
		discriminator_input->write_values(face_data->output(0)->value());		
		discriminator->forward();
		l2_loss_discriminator_mean->write_values(discriminator_output_mean->output(0)->value());
		l2_loss_discriminator_sigma->write_values(discriminator_output_sigma->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss_session->forward();		
		p_d_loss_avg.add(l2_loss->output(0)->value()->toFloat());
		std::cout << " - p_d_loss: " << p_d_loss_avg.result();
		l2_loss_session->backward();
		discriminator_output_mean->write_diffs(l2_loss_discriminator_mean->output(0)->diff());
		discriminator_output_sigma->write_diffs(l2_loss_discriminator_sigma->output(0)->diff());
		discriminator->backward();

		l2_loss->set_coef(1.0f);

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		generator_labels_output->forward();			
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_mean->write_values(discriminator_output_mean->output(0)->value());
		l2_loss_discriminator_sigma->write_values(discriminator_output_sigma->output(0)->value());
		l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
		l2_loss_session->forward();		
		n_d_loss_avg.add(l2_loss->output(0)->value()->toFloat());
		std::cout << " - n_d_loss: " << n_d_loss_avg.result();
		l2_loss_session->backward();
		discriminator_output_mean->write_diffs(l2_loss_discriminator_mean->output(0)->diff());
		discriminator_output_sigma->write_diffs(l2_loss_discriminator_sigma->output(0)->diff());
		discriminator->backward();			

		discriminator->apply_solvers();
				
		l2_loss->set_coef(1.0f);		

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		face_labels_node->forward();		
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_mean->write_values(discriminator_output_mean->output(0)->value());
		l2_loss_discriminator_sigma->write_values(discriminator_output_sigma->output(0)->value());
		l2_loss_labels_input->write_values(face_labels_node->output(0)->value());
		l2_loss_session->forward();
		float g_loss_1 = l2_loss->output(0)->value()->toFloat();
		g_loss_avg.add(g_loss_1);		
		l2_loss_session->backward();
		discriminator_output_mean->write_diffs(l2_loss_discriminator_mean->output(0)->diff());
		discriminator_output_sigma->write_diffs(l2_loss_discriminator_sigma->output(0)->diff());
		discriminator->backward();		
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		

		/*
		l2_loss->set_coef(-1.0f);

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		generator_labels_output->forward();
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_mean->write_values(discriminator_output_mean->output(0)->value());
		l2_loss_discriminator_sigma->write_values(discriminator_output_sigma->output(0)->value());
		l2_loss_labels_input->write_values(generator_labels_output->output(0)->value());
		l2_loss_session->forward();
		float g_loss_2 = l2_loss->output(0)->value()->toFloat();
		g_loss_avg.add(g_loss_2);
		l2_loss_session->backward();
		discriminator_output_mean->write_diffs(l2_loss_discriminator_mean->output(0)->diff());
		discriminator_output_sigma->write_diffs(l2_loss_discriminator_sigma->output(0)->diff());
		discriminator->backward();
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();
		*/
		std::cout << " - g_loss: " << g_loss_avg.result() << " | " << lr << " | " << sigma << std::endl;				

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

