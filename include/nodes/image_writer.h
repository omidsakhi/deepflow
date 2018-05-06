#pragma once

#include "core/node.h"

namespace cv {
	class Mat;
}

class DeepFlowDllExport ImageWriter : public Node {
public:
	ImageWriter(deepflow::NodeParam *param);
	int minNumInputs() { return 1; }
	int minNumOutputs() { return 0; }
	std::string op_name() const override { return "imwrite"; }
	void init();	
	void forward();
	void backward();
	std::string to_cpp() const;
	void set_text_line(std::string text);
	void set_filename(std::string filename);
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
	unsigned char * _output = nullptr;
	std::string _filename;
	std::string _text_line;
	std::shared_ptr<cv::Mat> disp;
};