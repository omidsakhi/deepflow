#include "core/node.h"
#include "core/common_cu.h"

__global__
void CpyAddKernel(const int n, const float alpha, const float *src, const float beta, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = beta * dst[i] + alpha * src[i];
}

__global__ 
void DotKernel(const int n, const float alpha, const float *a, const float *b, const float beta, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = beta * dst[i] + alpha * a[i] * b[i];
}

void Node::cpy(int n, const float alpha, const void * src, const float beta, void * dst, cudaStream_t stream)
{
	if (alpha == 1 && beta == 0) {
		DF_NODE_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else {
		CpyAddKernel << < numOfBlocks(n), maxThreadsPerBlock, 0, stream >> > (n, alpha, (float*)src, beta, (float*)dst);
		DF_KERNEL_CHECK();
	}
}

void Node::dot(const int n, const float alpha, const void *a, const void *b, const float beta, void *dst)
{
	DotKernel << < numOfBlocks(n), maxThreadsPerBlock >> > (n, alpha, (float*)a, (float*)b, beta, (float*)dst);
	DF_KERNEL_CHECK();
}

__global__
void NodeFillKernel(const int n, const float value, const float beta, float *dst)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = beta * dst[i] + value;
}
void Node::fill(int n, const float value, void * dst, const float beta, cudaStream_t stream)
{
	NodeFillKernel << < numOfBlocks(n), maxThreadsPerBlock, 0, stream >> > (n, value, beta, (float*) dst);
	DF_KERNEL_CHECK();
}

void Node::fill(const float value, cudaStream_t stream)
{
	fill(_outputs[0]->value()->size(), value, _outputs[0]->value()->mutableData(), 0.0f, stream);	
}


__global__
void GrayPictureGeneratorKernel(const int num_images, const float *in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char *out)
{
	int num_image = blockIdx.x*blockDim.x + threadIdx.x;
	if (num_image < num_images) {
		int input_width = per_image_height * per_image_width;

		float max = -FLT_MAX;
		float min = FLT_MIN;

		float tmp;
		for (int i = 0; i < input_width; i++) {
			tmp = in[num_image*input_width + i];
			if (max < tmp)
				max = tmp;
			if (min > tmp)
				min = tmp;
		}

		float denom = (max - min);
		if (denom == 0)
			denom = 1;

		int output_block_col = num_image % num_image_per_row_and_col;
		int output_block_row = (num_image - output_block_col) / num_image_per_row_and_col;
		int output_width = per_image_width * num_image_per_row_and_col;

		for (int i = 0; i < input_width; i++) {
			int input_image_col = i % per_image_width;
			int input_image_row = (i - input_image_col) / per_image_width;
			int output_col = output_block_col*per_image_width + input_image_col;
			int output_row = output_block_row*per_image_height + input_image_row;
			out[output_row*output_width + output_col] = (in[num_image*input_width + i] - min) / denom * 255;
		}
	}
}

__global__
void ColorPictureGeneratorKernel(const int num_images, const float *in, const int per_image_height, const int per_image_width, const int num_image_per_row_and_col, unsigned char *out)
{
	int num_image = blockIdx.x*blockDim.x + threadIdx.x;
	if (num_image < num_images) {
		int input_width = per_image_height * per_image_width;

		float max[3];
		float min[3];
		for (int i = 0; i < 3; i++) {
			max[i] = -FLT_MAX;
			min[i] = FLT_MAX;
		}

		float tmp[3];
		for (int i = 0; i < input_width; i++) {
			for (int j = 0; j < 3; j++) {
				tmp[j] = in[(3 * num_image + j)*input_width + i];
				if (max[j] < tmp[j]) max[j] = tmp[j];
				if (min[j] > tmp[j]) min[j] = tmp[j];
			}
		}

		float denom[3];
		for (int i = 0; i < 3; i++) {
			denom[i] = (max[i] - min[i]);
			if (denom[i] == 0)
				denom[i] = 1;
		}

		int output_block_col = num_image % num_image_per_row_and_col;
		int output_block_row = (num_image - output_block_col) / num_image_per_row_and_col;
		int output_width = per_image_width * num_image_per_row_and_col * 3;

		for (int i = 0; i < input_width; i++) {
			int input_image_col = i % per_image_width;
			for (int j = 0; j < 3; j++) {
				int output_image_row = (i - input_image_col) / per_image_width;
				int input_image_row = output_image_row * 3 + j;
				int output_col = 3 * (output_block_col*per_image_width + input_image_col) + j;
				int output_row = output_block_row*per_image_height + output_image_row;
				out[output_row*output_width + output_col] = (in[(3 * num_image + 2 - j)*input_width + i] - min[2 - j]) / denom[2 - j] * 255;
			}
		}

	}
}
