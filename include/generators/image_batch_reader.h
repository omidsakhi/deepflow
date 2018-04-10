#pragma once

#include "core/node.h"

#include <filesystem>


class Initializer;

#include <opencv2/opencv.hpp>

class DeepFlowDllExport ImageBatchReader : public Node {
public:
	ImageBatchReader(deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() override { return 1; }
	std::string op_name() const override { return "image_batch_reader"; }
	bool is_generator() override { return true; }
	void init() override;	
	void forward() override;
	void backward() override {}
	bool is_last_batch() override;
	std::string to_cpp() const override;
private:
	std::string _folder_path;
	std::array<int, 4> _dims;
	std::vector<int> _indices;
	std::vector<std::experimental::filesystem::path> _list_of_files;
	int _current_batch = -1;
	size_t _num_total_samples = 0;	
	int _num_batches = 0;
	int _batch_size = 0;
	bool _last_batch = false;	
	bool _randomize = false;
};
