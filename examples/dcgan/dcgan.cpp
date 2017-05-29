
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(model, "D:/Projects/deepflow/build/x64/Release/models/VGG_ILSVRC_16_layers.caffemodel", "Path to VGG16 model");
DEFINE_string(image_folder, "D:/Projects/deepflow/data/face", "Path to image folder dataset");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 12, "Batch size");
DEFINE_string(run,"", "Phase to execute graph");
DEFINE_bool(printiter, false, "Print iteration message");
DEFINE_bool(printepoch, true, "Print epoch message");
DEFINE_int32(debug, 0, "Level of debug");
DEFINE_int32(epoch, 1000, "Maximum epochs");
DEFINE_int32(iter, -1, "Maximum iterations");
DEFINE_bool(cpp, false, "Print C++ code");


void main(int argc, char** argv) {
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	CudaHelper::setOptimalThreadsPerBlock();	

	int batch_size = FLAGS_batch;
	
	DeepFlow df;
		

	if (FLAGS_i.empty()) {
		auto train = df.define_train_phase("Train");
		auto g_adam = df.adam_solver(0.0002f, 0.5f);
		auto d_adam = df.adam_solver(0.0001f, 0.5f);

		auto mean = 0;
		auto stddev = 0.02;

		auto gi = df.data_generator(df.random_normal({ FLAGS_batch, 1, 15, 6 }, mean, stddev, "random_input"), 60, "", "input", { train });

		auto gc1_f = df.variable(df.random_normal({ 1, 128, 5, 5 }, mean, stddev), g_adam, "gc1_f", { train });
		auto gc1 = df.transposed_conv2d(gi, gc1_f, 0, 0, 1, 1, 1, 1, "gc1", { train });
		auto gc2_f = df.variable(df.random_normal({ 128, 64, 5, 5 }, mean, stddev), g_adam, "gc2_f", { train });
		auto gc2 = df.transposed_conv2d(gc1, gc2_f, 0, 0, 1, 1, 1, 1, "gc2", { train });
		auto gc3_f = df.variable(df.random_normal({ 64, 1, 5, 5 }, mean, stddev), g_adam, "gc3_f", { train });
		auto gc3 = df.transposed_conv2d(gc2, gc3_f, 0, 0, 1, 1, 1, 1, "gc3", { train });

		auto di = df.image_batch_reader(FLAGS_image_folder, { FLAGS_batch, 1, 27, 18 }, true, "di", { train });
		
		auto select = df.data_generator(df.random_uniform({ 1,1,1,1 }, 0, 1.99), 60, "", "select");
		auto selector = df.multiplexer({ gc3, di }, select, "selector");
				
		auto dc1_f = df.variable(df.random_normal({ 64, 1, 5, 5 }, mean, stddev), d_adam, "dc1_f", { train });
		auto dc1 = df.conv2d(selector, dc1_f, 0, 0, 1, 1, 1, 1, "dc1", { train });
		auto dc2_f = df.variable(df.random_normal({ 128, 64, 5, 5 }, mean, stddev), d_adam, "dc2_f", { train });
		auto dc2 = df.conv2d(dc1, dc2_f, 0, 0, 1, 1, 1, 1, "dc2", { train });
		auto dc3_f = df.variable(df.random_normal({ 256, 128, 5, 5 }, mean, stddev), d_adam, "dc3_f", { train });
		auto dc3 = df.conv2d(dc2, dc3_f, 0, 0, 1, 1, 1, 1, "dc3", { train });		
		
		auto disp = df.display(dc3);
		
	}
	else {
		df.block()->load_from_binary(FLAGS_i);	
	}
	
	auto session = df.session();	

	if (!FLAGS_run.empty()) {
		session->initialize();
		session->run(FLAGS_run, FLAGS_epoch, FLAGS_iter, FLAGS_printiter, FLAGS_printepoch, FLAGS_debug);
	}

	if (FLAGS_cpp) {
		session->initialize();
		std::cout << session->to_cpp() << std::endl;
	}

	
	if (!FLAGS_o.empty())
	{
		if (FLAGS_text)
			df.block()->save_as_text(FLAGS_o);
		else
			df.block()->save_as_binary(FLAGS_o);
	}
	
}

