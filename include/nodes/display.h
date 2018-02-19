#pragma once

#include "core/node.h"

#include <opencv2/opencv.hpp>

class DeepFlowDllExport Display : public Node {
public:
	Display(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 1; }	
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
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
	int _delay_msec = 100;
	int _epoch_frequency = 1;
	deepflow::ActionType _display_type;
	deepflow::ActionTime _display_time;
};
