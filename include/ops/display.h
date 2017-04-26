#pragma once

#include "core/node.h"

#include <opencv2/opencv.hpp>

class DeepFlowDllExport Display : public Node {
public:
	Display(const NodeParam &param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 0; }
	void initForward();
	void initBackward();
	void forward();
	void backward();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return NEVER_BACKWARD; }
protected:
	int input_size;
	int input_size_in_bytes;
	int num_images;
	int per_image_height;
	int per_image_width;
	int num_image_per_row_and_col;
	int pic_width;
	int pic_height;
	int num_pic_pixels;
	unsigned char *d_pic;	
	cv::Mat disp;
	int _delay_msec = 100;
	DisplayParam_DisplayType _display_type;
};
