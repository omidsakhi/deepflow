#pragma once

#include "nodes/variable.h"

class Initializer;

namespace cv {
	class Mat;
}

class DeepFlowDllExport TextImageGenerator : public Variable {
public:
	TextImageGenerator(std::shared_ptr<Initializer> initializer, deepflow::NodeParam *param);
	int minNumInputs() override { return 0; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "text_image_generator"; }
	bool is_generator() override;
	void init() override;
	void forward() override;
	std::string to_cpp() const override;
private:
	int _n = 0;
	int _c = 0;
	int _h = 0;
	int _w = 0;
	bool _no_solver = false;
	std::string _dict;
	std::shared_ptr<cv::Mat> cv_img;
	unsigned char *d_img = nullptr;
};
