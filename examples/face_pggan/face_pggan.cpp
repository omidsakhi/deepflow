
#include "core/deep_flow.h"
#include "core/session.h"

#include <random>
#include <gflags/gflags.h>

DEFINE_string(faces, "C:/Projects/deepflow/data/celeba128", "Path to face dataset folder");
DEFINE_int32(channels, 3, "Image channels");
DEFINE_int32(size, 128, "Image size");
DEFINE_int32(total, 2000000, "Image channels"); // 13230
DEFINE_int32(batch, 20, "Batch size");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(iter, 10000, "Maximum iterations");
DEFINE_string(load, "", "Load from gXXXX.bin and dXXXX.bin");
DEFINE_int32(save_image, 100, "Save image iter frequency (Don't Save = 0)");
DEFINE_int32(save_model, 1000, "Save model iter frequency (Don't Save = 0)");

std::shared_ptr<Session> load_session(std::string suffix) {
	std::string filename = "s" + suffix + ".bin";
	std::cout << "Loading session from " << filename << std::endl;
	DeepFlow df;
	df.block()->load_from_binary(filename);
	return df.session();
}

std::string deconv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, std::string name) {

	auto node = df->transposed_conv2d(input, input_channels, output_channels, kernel, pad, 2, false, solver, name + "_tconv");
	return df->batch_normalization(node, solver, output_channels, 0.001f, name + "_tconv_bn");
}

std::string to_rgb(DeepFlow *df, std::string input, std::string solver, int input_channels, std::string name) {
	return df->conv2d(input, input_channels, 3, 1, 0, 1, true, solver, name + "_conv");
}

std::string from_rgb(DeepFlow *df, std::string input, std::string solver, int output_channels, std::string name) {
	return df->conv2d(input, 3, output_channels, 1, 0, 1, true, solver, name + "_conv");
}

std::string conv(DeepFlow *df, std::string input, std::string solver, int input_channels, int output_channels, int kernel, int pad, bool activation, std::string name) {
	auto node = df->conv2d(input, input_channels, output_channels, kernel, pad, 2, false, solver, name + "_conv");
	node = df->batch_normalization(node, solver, output_channels, 0.001f, name + "_conv_bn");
	if (activation) {
		return df->prelu(node, output_channels, solver, name + "_relu");
	}
	return node;
}

std::shared_ptr<Session> create() {
	DeepFlow df;

	int fn = 256;	
	
	auto g_solver = df.adam_solver(0.000051f, 0.5f, 0.98f, 10e-8f, "g_adam");	
	auto d_solver = df.adam_solver(0.00005f, 0.5f, 0.98f, 10e-8f, "d_adam");

	// The input for generator when needed
	auto z = df.data_generator(df.random_normal({ FLAGS_batch, 100, 1, 1 }, 0, 1), "", "z");	
	
	auto face_data_128 = df.image_batch_reader(FLAGS_faces, { FLAGS_batch, FLAGS_channels, FLAGS_size, FLAGS_size }, false, "face_data_128");
	auto face_data_64 = df.resize(face_data_128, 0.5, 0.5, "face_data_64");
	auto face_data_32 = df.resize(face_data_64, 0.5, 0.5, "face_data_32");
	auto face_data_16 = df.resize(face_data_32, 0.5, 0.5, "face_data_16");
	auto face_data_8 = df.resize(face_data_16, 0.5, 0.5, "face_data_8");
	
	auto face_labels = df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 1), "", "face_labels");
	auto generator_labels = df.data_generator(df.fill({ FLAGS_batch, 1, 1, 1 }, 0), "", "generator_labels");

	auto g4 = df.dense(z, { 100, fn , 4 , 4 }, true, g_solver, "g4"); // 4x4

	auto g8 = deconv(&df, g4, g_solver, fn, fn, 3, 1, "g8"); // 8x8
	auto g8elu = df.leaky_relu(g8, 0.2, "g8elu");
	auto g8trgb = to_rgb(&df, g8, g_solver, fn, "g8trgb");

	auto g16 = deconv(&df, g8elu, g_solver, fn, fn, 3, 1, "g16"); // 16x16
	auto g16elu = df.leaky_relu(g16, 0.2, "g16elu");
	auto g16trgb = to_rgb(&df, g16, g_solver, fn, "g16trgb");
	auto g16add = df.add(df.resize(g8trgb, 2, 2), g16trgb, "g16add");
	auto im16 = df.switcher(g16trgb, "im16");
	df.imwrite(im16, "i16-{it}", "imw16");

	auto g32 = deconv(&df, g16elu, g_solver, fn, fn, 3, 1, "g32"); // 32x32
	auto g32elu = df.leaky_relu(g32, 0.2, "g32elu");
	auto g32trgb = to_rgb(&df, g32, g_solver, fn, "g32trgb");
	auto g32add = df.add(df.resize(g16add, 2, 2), g32trgb, "g32add");
	auto im32 = df.switcher(g32trgb, "im32");
	df.imwrite(im32, "i32-{it}", "imw32");

	auto g64 = deconv(&df, g32elu, g_solver, fn, fn, 3, 1, "g64"); // 64x64
	auto g64elu = df.leaky_relu(g64, 0.2, "g64elu");
	auto g64trgb = to_rgb(&df, g64, g_solver, fn, "g64trgb");
	auto g64add = df.add(df.resize(g32add, 2, 2), g64trgb, "g64add");
	auto im64 = df.switcher(g64trgb, "im64");
	df.imwrite(im64, "i64-{it}", "imw64");

	auto g128 = deconv(&df, g64elu, g_solver, fn, fn, 3, 1, "g128"); // 64x64 -> 128x128
	auto g128trgb = to_rgb(&df, g128, g_solver, fn, "g128trgb");
	auto g128add = df.add(df.resize(g64add, 2, 2), g128trgb, "g128add");
	auto im128 = df.switcher(g128trgb, "im128");
	df.imwrite(im128, "i128-{it}", "imw128");

	auto m128 = df.multiplexer({ face_data_128, g128add }, "m128");
	auto m128frgb = from_rgb(&df, m128, d_solver, fn, "m128frgb");
	
	auto d128 = conv(&df, m128frgb, d_solver, fn, fn, 3, 1, true, "d128"); // 128x128 -> 64x64
	auto m64 = df.multiplexer({ face_data_64, g64add }, "m64");
	auto m64frgb = from_rgb(&df, m64, d_solver, fn, "m64frgb");
	auto m642 = df.multiplexer({ d128, m64frgb }, "m642");

	auto d64 = conv(&df, m642, d_solver, fn, fn, 3, 1, true, "d64"); // 64x64 -> 32x32
	auto m32 = df.multiplexer({ face_data_32, g32add }, "m32");
	auto m32frgb = from_rgb(&df, m32, d_solver, fn, "m32frgb");
	auto m322 = df.multiplexer({ d64, m32frgb }, "m322");

	auto d32 = conv(&df, m322, d_solver, fn, fn, 3, 1, true, "d32"); // 32x32 -> 16x16
	auto m16 = df.multiplexer({ face_data_16, g16add }, "m16");
	auto m16frgb = from_rgb(&df, m16, d_solver, fn, "m16frgb");
	auto m162 = df.multiplexer({ d32, m16frgb }, "m162");

	auto d16 = conv(&df, m162, d_solver, fn, fn, 3, 1, true, "d16"); // 16x16 -> 8x8
	auto m8 = df.multiplexer({ face_data_8, g8trgb }, "m8");
	auto m8frgb = from_rgb(&df, m8, d_solver, fn, "m8frgb");
	auto m82 = df.multiplexer({ d16, m8frgb }, "m82");

	auto d8 = df.dense(m82, { fn * 8 * 8, 1, 1, 1 }, true, d_solver, "d8");	

	auto mloss = df.multiplexer({ face_labels, generator_labels }, "mloss");
	
	std::string err;
	err = df.reduce_mean(df.square_error(d8, mloss));
	auto loss = df.loss(err, DeepFlow::AVG);

	return df.session();
}

void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();

	auto execution_context = std::make_shared<ExecutionContext>();
	execution_context->debug_level = FLAGS_debug;

	std::shared_ptr<Session> session;
	if (FLAGS_load.empty())
		session = create();
	else
		session = load_session(FLAGS_load);
	session->initialize(execution_context);
	
	// m82, m162, m322, m642
	int main_multiplex_states[5][4] = {
		1, -1, -1, -1, // stage 0
		0,  1, -1, -1, // stage 1
		0,  0,  1, -1, // stage 2
		0,  0,  0,  1, // stage 3
		0,  0,  0,  0  // stage 4
	};

	std::shared_ptr<Add> add_nodes[4] = {
		std::dynamic_pointer_cast<Add>(session->get_node("g16add")),
		std::dynamic_pointer_cast<Add>(session->get_node("g32add")),
		std::dynamic_pointer_cast<Add>(session->get_node("g64add")),
		std::dynamic_pointer_cast<Add>(session->get_node("g128add"))
	};

	std::shared_ptr<Multiplexer> main_multiplex[4] = {
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m82")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m162")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m322")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m642"))
	};

	std::shared_ptr<Multiplexer> per_stage_multiplex[6] = {
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m8")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m16")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m32")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m64")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("m128")),
		std::dynamic_pointer_cast<Multiplexer>(session->get_node("mloss"))
	};
	
	std::shared_ptr<Switch> im_switches[5] = {		
		std::dynamic_pointer_cast<Switch>(session->get_node("im16")),
		std::dynamic_pointer_cast<Switch>(session->get_node("im32")),
		std::dynamic_pointer_cast<Switch>(session->get_node("im64")),
		std::dynamic_pointer_cast<Switch>(session->get_node("im128"))
	};

	auto loss = session->get_node("loss");

	int iter = 1;	
	
	int stage;

	for (stage = 1; stage < 5; stage++) {
		for (iter = 1; iter <= FLAGS_iter && execution_context->quit != true; ++iter) {			
				execution_context->current_iteration = iter;
				
				/*
				if (stage > 0) {
					float beta = (float)iter / 1000.0f;
					if (beta > 1) beta = 1;					
					add_nodes[stage - 1]->setAlpha(1 - beta);
					add_nodes[stage - 1]->setBeta(beta);
				}
				*/

				std::cout << "Stage " << stage << " Iteration: [" << iter << "/" << FLAGS_iter << "]";
				for (int m = 0; m < 4; m++)
					main_multiplex[m]->selectInput(main_multiplex_states[stage][m]);

				for (int s = 0; s < 4; s++)
					im_switches[s]->setEnabled(false);

				for (int m = 0; m < 6; m++)
					per_stage_multiplex[m]->selectInput(-1);
				per_stage_multiplex[5]->selectInput(0);
				per_stage_multiplex[stage]->selectInput(0);

				session->forward( session->end_nodes() );
				float d_loss = loss->output(0)->value()->toFloat();
				session->backward( session->end_nodes() );

				for (int m = 0; m < 6; m++)
					per_stage_multiplex[m]->selectInput(-1);
				per_stage_multiplex[5]->selectInput(1);
				per_stage_multiplex[stage]->selectInput(1);

				session->forward(session->end_nodes() );
				d_loss += loss->output(0)->value()->toFloat();
				session->backward(session->end_nodes() );

				std::cout << " - d_loss: " << d_loss;

				session->apply_solvers({ "d_adam" });
				session->reset_gradients();

				if (FLAGS_save_image != 0 && iter % FLAGS_save_image == 0 && stage > 0)
					im_switches[stage - 1]->setEnabled(true);

				for (int m = 0; m < 6; m++)
					per_stage_multiplex[m]->selectInput(-1);
				per_stage_multiplex[5]->selectInput(0);
				per_stage_multiplex[stage]->selectInput(1);

				session->forward(session->end_nodes() );
				float g_loss = loss->output(0)->value()->toFloat();
				session->backward(session->end_nodes() );
				std::cout << " - g_loss: " << g_loss << std::endl;
				session->apply_solvers({ "g_adam" });
				session->reset_gradients();

				if (FLAGS_save_model != 0 && iter % FLAGS_save_model == 0) {
					session->save("model_s" + std::to_string(stage) + "_i" + std::to_string(iter) + ".bin");
				}
			}
		}
			
}

