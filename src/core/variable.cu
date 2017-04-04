
#include "core/variable.h"
#include "core/initializer.h"

#include <string>
#include <iostream>

#include "core/common_cu.h"

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

Variable::Variable(std::shared_ptr<Initializer> initializer, NodeParam param) : Node(param) {
	LOG_IF(FATAL, param.has_variable_param() == false) << "param.has_variable_param() == false";
	_initializer = initializer;			
}

void Variable::initForward() {	
	_initializer->init();
	_outputs[0]->initValue(_initializer->dims());	
	LOG(INFO) << "Initialize Variable (name: " << _name << ") - " << _outputs[0]->value()->toString();
	_initializer->apply(this);
}

void Variable::forward() {
	
}

void Variable::initBackward() {	
	_outputs[0]->initDiff();
}

bool Variable::snapshot() {
	return _param.variable_param().has_snapshot_param();
}

int Variable::snapshotInterval() {
	return _param.variable_param().snapshot_param().snapshot_interval();
}


void Variable::toImage(int iteration) {	
	LOG_IF(FATAL, _param.variable_param().has_snapshot_param() == false);
	const SnapshotParam &param = _param.variable_param().snapshot_param();
	int perImageWidth = param.per_image_width();
	int perImageHeight = param.per_image_height();
	std::string imagePath = param.snapshot_prefix() + std::to_string(iteration) + ".png";
	LOG(INFO) << _name << " to image " << imagePath;	
	auto weights = _outputs[0]->value()->cpyToHost<float>();
	auto dims = _outputs[0]->value()->dims();
	int numImages = dims[1];
	int perImage = dims[0];
	int sq = (int)floor(sqrt((float)numImages));
	int picWidth = perImageWidth * sq;
	int picHeight = perImageHeight * ((int)ceil(((float)numImages / sq)));
	for (int c = 0; c < numImages; ++c)
	{
		float max = -FLT_MAX;
		float min =  FLT_MAX;
		for (int r = 0; r < perImage; ++r)
		{
			float tmp = weights->at(r*numImages + c);
			if (max < tmp)
				max = tmp;
			if (min > tmp)
				min = tmp;
		}
		for (int r = 0; r < perImage; ++r)
		{
			float &tmp = weights->at(r*numImages + c);
			tmp = (tmp - min) / (max-min) * 255;
		}
	}
	cv::Mat result(picHeight, picWidth, CV_32SC1);
	result = 0;
	for (int c = 0; c < numImages; ++c)
	{
		for (int r = 0; r < perImage; ++r)
		{
			float &tmp = weights->at(r*numImages + c);
			int csmall = r % perImageWidth;
			int rsmall = (r - csmall) / perImageWidth;
			int cbig = c % sq;
			int rbig = (c - cbig) / sq;
			int nc = cbig*perImageWidth + csmall;
			int nr = rbig*perImageHeight + rsmall;
			result.at<int>(nr, nc) = tmp;
		}
	}
	cv::imwrite(imagePath, result);
}

/*
__global__
void PlaceHolderImageKernel(const int n,const float *in, int *out, const int picWidth, const int sq, const int numImages, const int perImageWidth, const int perImageHeight)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < n) {
int flat_pixel = i % (perImageWidth * perImageHeight);
int num_image = (i-flat_pixel) / (perImageWidth * perImageHeight);
int csmall = flat_pixel % perImageWidth;
int rsmall = (flat_pixel - csmall) / perImageWidth;
int cbig = num_image % sq;
int rbig = (num_image - cbig) / sq;
int c = cbig*perImageWidth + csmall;
int r = rbig*perImageHeight + rsmall;
out[r*picWidth+c] = (in[i] + 1.0f) / 2.0f * 255;
}
}

void PlaceHolder::toImage(std::string imagePath) {
LOG(INFO) << _name << " to image " << imagePath;
auto shape = _outputs[0]->shape();
int numImages = shape->at(1);
int perImageHeight = shape->at(2);
int perImageWidth = shape->at(3);
int sq = (int)floor(sqrt((float)numImages));
int picWidth = perImageWidth * sq;
int picHeight = perImageHeight * ((int)ceil(((float)numImages / sq)));
int n = picWidth * picHeight;
int *d_img;
LOG_IF(FATAL, cudaMalloc(&d_img, sizeof(int) * n) != 0);
LOG_IF(FATAL, cudaMemset(d_img, 0, sizeof(int) * n) != 0);
PlaceHolderImageKernel << < numOfBlocks(shape->size()), maxThreadsPerBlock >> >(shape->size(), _outputs[0]->value(), d_img, picWidth, sq, numImages,perImageWidth, perImageHeight);
LOG_IF(FATAL, cudaPeekAtLastError() != 0);
int *img = new int[n];
LOG_IF(FATAL, cudaMemcpy(img, d_img, sizeof(int) *n,cudaMemcpyDeviceToHost) != 0);
LOG_IF(FATAL, cudaFree(d_img) != 0);
cv::Mat imgMat(picHeight,picWidth,CV_32SC1,img);
cv::imwrite(imagePath, imgMat);
delete[] img;
}

*/