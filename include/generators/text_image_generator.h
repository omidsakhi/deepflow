#pragma once

#include "core/node.h"

class Initializer;

namespace cv {
	class Mat;
}

class DeepFlowDllExport TextImageGenerator : public Node {
public:
	TextImageGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() { return 2; }
	std::string op_name() const override { return "text_image_generator"; }
	bool is_generator() override;
	void init() override;
	void forward() override;
	void backward() override {};
	std::string to_cpp() const override;
private:
	std::string generate_random_text(int max_characters, std::string &dict);
private:
	std::shared_ptr<Initializer> _initializer;
	int _n = 0;
	int _c = 0;
	int _h = 0;
	int _w = 0;
	bool _no_solver = false;
	std::string _chars;	
	std::shared_ptr<cv::Mat> cv_img_actual, cv_img_target;
	unsigned char *d_img_actual = nullptr;
	unsigned char *d_img_target = nullptr;
};