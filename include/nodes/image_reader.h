#pragma once

#include "core/node.h"

#include <filesystem>


class Initializer;

#include <opencv2/opencv.hpp>

class DeepFlowDllExport ImageReader : public Node {
public:
	enum ColorType {
		GRAY_ONLY,
		COLOR_IF_AVAILABLE
	};
	ImageReader(deepflow::NodeParam *param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	std::string op_name() const override { return "imread"; }
	void init();	
	void forward() {}
	void backward() {}
	std::string to_cpp() const;
private:		
	cv::Mat img;
};
