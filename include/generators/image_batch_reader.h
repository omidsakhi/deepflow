#pragma once


#include "core/generator.h"
#include "core/node.h"

#include <filesystem>


class Initializer;

#include <opencv2/opencv.hpp>

class DeepFlowDllExport ImageBatchReader : public Generator, public Node {
public:
	ImageBatchReader(const deepflow::NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward() {}
	bool isLastBatch();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	std::string _folder_path;
	std::array<int, 4> _dims;
	std::vector<std::experimental::filesystem::path> _list_of_files;
	int _current_batch = 0;
	size_t _num_total_samples;
	size_t _num_batch_samples;
	bool _last_batch;	
	cv::Mat img;
	unsigned char *d_img;	
};
