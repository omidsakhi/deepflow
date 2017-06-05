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
	ImageReader(const deepflow::NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }	
	void initForward();
	void initBackward() {}	
	std::string to_cpp() const;
	ForwardType forwardType() { return ALWAYS_FORWARD; }
	BackwardType backwardType() { return NEVER_BACKWARD; }
private:		
	cv::Mat img;
};
