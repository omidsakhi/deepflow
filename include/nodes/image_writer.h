#pragma once

#include "core/node.h"

#include <opencv2/opencv.hpp>

class DeepFlowDllExport ImageWriter : public Node {
public:
	ImageWriter(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:
	int input_size;
	int input_size_in_bytes;
	int num_images;
	int num_samples;
	int num_channels;
	int per_image_height;
	int per_image_width;
	int num_image_per_row_and_col;
	int pic_width;
	int pic_height;
	int num_pic_pixels;
	cv::Mat disp;
};