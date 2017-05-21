
#include "core/deep_flow.h"
#include "core/session.h"

#include <gflags/gflags.h>

DEFINE_string(model, "D:/Projects/deepflow/build/x64/Release/models/VGG_ILSVRC_16_layers.caffemodel", "Path to VGG16 model");
DEFINE_string(i, "", "Trained network model to load");
DEFINE_string(o, "", "Trained network model to save");
DEFINE_bool(text, false, "Save model as text");
DEFINE_bool(includeweights, false, "Also save weights in text mode");
DEFINE_bool(includeinits, false, "Also save initial values");
DEFINE_int32(batch, 100, "Batch size");
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
		df.load_from_caffe_model(FLAGS_model, { std::pair<std::string, std::array<int,4>>("data",{ 4,3,224,224 }) }, FLAGS_debug > 0);
		df.block()->remove_node_params({ "fc6_ip", "fc6_w", "fc6_b", "fc7_w", "fc7_b", "fc8_w", "fc8_b" });
		auto train = df.define_train_phase("Train");
		auto solver = df.gain_solver(1.0f, 0.01f, 100, 0.000000001f, 0.05f, 0.95f);
		df.block()->set_solver_for_variable_params(solver, {});
		df.block()->set_phase_for_node_params(train, {});
	}
	else {
		df.block()->load_from_binary(FLAGS_i);	
	}
	
	df.block()->print_node_params();
	df.block()->print_phase_params();

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

