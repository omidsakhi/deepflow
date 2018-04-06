
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

std::shared_ptr<Session> create_z() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 100, 1, 1 }, -1, 1),"", "z");	
	return df.session();
}

std::shared_ptr<Session> create_static_z() {
	DeepFlow df;
	df.data_generator(df.random_uniform({ FLAGS_batch, 100, 1, 1 }, -1, 1),"", "static_z");
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
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), "", "face_labels");
	return df.session();
}

std::shared_ptr<Session> create_generator_labels() {
	DeepFlow df;
	df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), "", "generator_labels");
	return df.session();
}

std::shared_ptr<Session> create_l2_loss_1() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");	
	auto discriminator_target = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_target");
	std::string err;
	err = df.reduce_mean(df.square_error(discriminator_input, discriminator_target));
	auto loss = df.loss(err, DeepFlow::AVG);
	df.print({ loss }, "", DeepFlow::END_OF_EPOCH);
	return df.session();
}

std::shared_ptr<Session> create_l2_loss_2() {
	DeepFlow df;
	auto discriminator_input = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_input");
	auto discriminator_target = df.place_holder({ FLAGS_batch, 1, 1, 1 }, Tensor::Float, "discriminator_target");
	std::string err;
	err = df.square_error(discriminator_input, discriminator_target);
	auto loss = df.loss(err, DeepFlow::AVG);
	df.print({ loss }, "", DeepFlow::END_OF_EPOCH);
	return df.session();
}

std::shared_ptr<Session> create_generator() {
	DeepFlow df;
	
	int fn = 64;

	auto solver = df.adam_solver(0.0002f, 0.5f, 0.99f);
	
	auto node = df.place_holder({ FLAGS_batch, 100, 1, 1 }, Tensor::Float, "input");

	node = df.dense(node, { 100, fn * 4, 4 , 4 }, false, solver, "gfc");
	node = df.batch_normalization(node, solver, fn * 4, "gfc_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 2, false, solver, "g8_1");
	node = df.batch_normalization(node, solver, fn * 4, "g8_1_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 1, false, solver, "g8_2");
	node = df.batch_normalization(node, solver, fn * 4, "g8_2_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 2, false, solver, "g16_1");
	node = df.batch_normalization(node, solver, fn * 4, "g16_1_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 4, fn * 4, 3, 1, 1, false, solver, "g16_2");
	node = df.batch_normalization(node, solver, fn * 4, "g16_2_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 4, fn * 2, 3, 1, 2, false, solver, "g32_1");
	node = df.batch_normalization(node, solver, fn * 2, "g32_1_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 2, fn * 2, 3, 1, 1, false, solver, "g32_2");
	node = df.batch_normalization(node, solver, fn * 2, "g32_2_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 2, fn * 2, 3, 1, 2, false, solver, "g64_1");
	node = df.batch_normalization(node, solver, fn * 2, "g64_1_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn * 2, fn, 3, 1, 2, false, solver, "g128_1");
	node = df.batch_normalization(node, solver, fn, "g128_1_bn");
	node = df.relu(node);

	node = df.transposed_conv2d(node, fn, 3, 3, 1, 1, false, solver, "g128_2");
	//node = df.bias_add(node, 3, solver, "g128_2_b");
	//node = df.tanh(node);

	return df.session();

}

std::shared_ptr<Session> create_discriminator() {
	DeepFlow df;

	int fn = 64;

	auto solver = df.adam_solver(0.0002f, 0.5f, 0.99f);

	auto node = df.place_holder({ FLAGS_batch , FLAGS_channels, FLAGS_size, FLAGS_size }, Tensor::Float, "input");	

	node = df.conv2d(node, 3, fn, 5, 2, 2, false, solver, "d64");
	node = df.leaky_relu(node, 0.2f);

	node = df.conv2d(node, fn, fn * 2, 5, 2, 2, false, solver, "d32");
	node = df.batch_normalization(node, solver, fn * 2, "d32_bn");
	node = df.leaky_relu(node, 0.2f);

	node = df.conv2d(node, fn * 2, fn * 4, 5, 2, 2, false, solver, "d16");
	node = df.batch_normalization(node, solver, fn * 4, "d16_bn");
	node = df.leaky_relu(node, 0.2f);

	node = df.conv2d(node, fn * 4, fn * 4, 5, 2, 2, false, solver, "d8");
	node = df.batch_normalization(node, solver, fn * 4, "d8_bn");
	node = df.leaky_relu(node, 0.2f);

	node = df.conv2d(node, fn * 4, fn * 8, 5, 2, 2, false, solver, "d4");
	node = df.batch_normalization(node, solver, fn * 8, "d4_bn");
	node = df.leaky_relu(node, 0.2f);

	df.dense(node, { (fn * 8) * 4 * 4, 1, 1, 1 }, true, solver, "dfc");

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

	auto l2_loss_session_1 = create_l2_loss_1();
	l2_loss_session_1->initialize(execution_context);	

	auto l2_loss_session_2 = create_l2_loss_2();
	l2_loss_session_2->initialize(execution_context);

	auto z_session = create_z();
	z_session->initialize(execution_context);

	auto static_z_session = create_static_z();
	static_z_session->initialize(execution_context);

	auto face_data = face_reader_session->get_node("face_data");	
	auto generator_output = generator->end_node();
	auto generator_input = generator->get_placeholder("input");
	auto discriminator_input = discriminator->get_placeholder("input");	
	auto discriminator_output = discriminator->end_node();	
	auto generator_labels_output = generator_labels->get_node("generator_labels");
	auto face_labels_node = face_labels->get_node("face_labels");
	auto l2_loss_discriminator_input_1 = l2_loss_session_1->get_placeholder("discriminator_input");	
	auto l2_loss_discriminator_target_1 = l2_loss_session_1->get_placeholder("discriminator_target");
	auto l2_loss_1 = std::dynamic_pointer_cast<Loss>(l2_loss_session_1->get_node("loss"));	
	auto l2_loss_discriminator_input_2 = l2_loss_session_2->get_placeholder("discriminator_input");
	auto l2_loss_discriminator_target_2 = l2_loss_session_2->get_placeholder("discriminator_target");
	auto l2_loss_2 = std::dynamic_pointer_cast<Loss>(l2_loss_session_2->get_node("loss"));
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
	
	for ( ; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {		

		execution_context->current_iteration = iter;
		std::cout << "Iteration: [" << iter << "/" << FLAGS_iter << "]";

		l2_loss_1->set_alpha(0.01f);
		l2_loss_1->set_beta(0.99f);

		face_reader_session->forward();
		face_labels_node->forward();		
		discriminator_input->write_values(face_data->output(0)->value());		
		discriminator->forward();
		l2_loss_discriminator_input_1->write_values(discriminator_output->output(0)->value());		
		l2_loss_discriminator_target_1->write_values(face_labels_node->output(0)->value());
		l2_loss_session_1->forward();		
		p_d_loss_avg.add(l2_loss_1->output(0)->value()->toFloat());
		std::cout << " - p_d_loss: " << p_d_loss_avg.result();
		l2_loss_session_1->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input_1->output(0)->diff());		
		discriminator->backward();

		l2_loss_1->set_alpha(0.01f);
		l2_loss_1->set_beta(0.99f);

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		generator_labels_output->forward();			
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input_1->write_values(discriminator_output->output(0)->value());		
		l2_loss_discriminator_target_1->write_values(generator_labels_output->output(0)->value());
		l2_loss_session_1->forward();		
		n_d_loss_avg.add(l2_loss_1->output(0)->value()->toFloat());
		std::cout << " - n_d_loss: " << n_d_loss_avg.result();
		l2_loss_session_1->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input_1->output(0)->diff());		
		discriminator->backward();			

		discriminator->apply_solvers();
				
		l2_loss_2->set_alpha(0.01f);		
		l2_loss_2->set_beta(0.99f);

		z->forward();
		generator_input->write_values(z->output(0)->value());
		generator->forward();
		face_labels_node->forward();		
		discriminator_input->write_values(generator_output->output(0)->value());
		discriminator->forward();
		l2_loss_discriminator_input_2->write_values(discriminator_output->output(0)->value());		
		l2_loss_discriminator_target_2->write_values(face_labels_node->output(0)->value());
		l2_loss_session_2->forward();
		float g_loss_1 = l2_loss_2->output(0)->value()->toFloat();
		g_loss_avg.add(g_loss_1);		
		l2_loss_session_2->backward();
		discriminator_output->write_diffs(l2_loss_discriminator_input_2->output(0)->diff());		
		discriminator->backward();		
		generator_output->write_diffs(discriminator_input->output(0)->diff());
		generator->backward();

		std::cout << " - g_loss: " << g_loss_avg.result() << std::endl;				

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

