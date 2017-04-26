#pragma once


#include "core/generator.h"
#include "core/node.h"

class Initializer;

#include <opencv2/opencv.hpp>

class DeepFlowDllExport ImageReader : public Generator, public Node {
public:
	enum ColorType {
		GRAY_ONLY,
		COLOR_IF_AVAILABLE
	};
	ImageReader(const NodeParam &param);
	int minNumInputs() { return 0; }
	int minNumOutputs() { return 1; }
	void nextBatch();
	void initForward();
	void initBackward();
	void forward();
	void backward();
	bool isLastBatch();
	virtual ForwardType forwardType() { return ALWAYS_FORWARD; }
	virtual BackwardType backwardType() { return NEVER_BACKWARD; }
private:	
	bool _last_batch = true;
	cv::Mat img;
};
