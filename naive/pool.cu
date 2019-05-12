#include "fashion.h"


void Pool::init(int minib, int Inputimage_h, int Inputimage_w, int Inputimage_ch, int pool_size)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0,1.0);

	this->Inputimage_height=Inputimage_h;
	this->Inputimage_width=Inputimage_w;
	this->Inputimage_channel=Inputimage_ch;
	this->Outputimage_height=Inputimage_h/pool_size;
	this->Outputimage_width=Inputimage_w/pool_size;
	this->Outputimage_channel=Inputimage_ch;
	this->Output_height=minib;
	this->Output_width=Outputimage_channel*Outputimage_height*Outputimage_width;
	this->minibatch=minib;
	this->X_height=minib;
	this->X_width=Inputimage_channel*Inputimage_height*Inputimage_width;
	this->b_height=minib;
	this->b_width=Inputimage_channel;
	this->pool_size=pool_size;

	this->X_c.resize(minibatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
	this->X.resize(minibatch*Inputimage_channel*Inputimage_height*Inputimage_width, 0);
	this->Output_c.resize(minibatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
	this->Output.resize(minibatch*Outputimage_channel*Outputimage_height*Outputimage_width, 0);
	this->b.resize(Inputimage_channel,0.1);
	this->b_c.resize(Inputimage_channel,0.1);
}

void Pool::forward_GPU_naive(device_vector<float> & input)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	int bz = ceil((float)Outputimage_width/TILE_WIDTH)*ceil((float)Outputimage_height/TILE_WIDTH);
	if( bz == 0 )
		bz = 1;
	dim3 numBlocks(minibatch, Outputimage_channel, bz);

	float* input_pointer = thrust::raw_pointer_cast( input.data() );
	float* Output_pointer = thrust::raw_pointer_cast( Output.data() );
	poolingLayer_forward_GPU_naive<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
			Inputimage_width, Output_pointer, Outputimage_channel, pool_size);

}

//double for loop version
void Pool::backward_GPU(device_vector<float> & output)
{
	dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH);
	int bz = ceil((float)Outputimage_width/TILE_WIDTH)*ceil((float)Outputimage_height/TILE_WIDTH);
	if( bz == 0 )
		bz = 1;
	dim3 numBlocks(minibatch, Outputimage_channel, bz);

	float* input_pointer = thrust::raw_pointer_cast( X.data() );
	float* output_pointer = thrust::raw_pointer_cast( output.data() );

	poolingLayer_backward_GPU<<<numBlocks,threadsPerBlock>>>(input_pointer, Inputimage_height,
			Inputimage_width, output_pointer, Outputimage_channel, pool_size);
}

//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
__global__ void poolingLayer_forward_GPU_naive(float* X, int H_in, int W_in, float* Y, int M, int pool_size)
{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;
	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				acc = acc + X[n*(M*H_in*W_in)+ m*(H_in*W_in) +
				              (pool_size * h + p)*(W_in) + (pool_size * w + q)] / (pool_size * pool_size);
	}
	__syncthreads();
	if(h < H_out && w < W_out)
	{
		Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w] = acc;
	}
}

//double for loop version
//M: number of input, output feature maps
//H_in: height of each input image
//W_in: width of each input map image
//X: input feature maps
//Y: output feature maps
__global__ void poolingLayer_backward_GPU(float* X, int H_in, int W_in, float* Y, int M, int pool_size)
{
	int n, m, h, w, p, q;
	int H_out = H_in/pool_size;
	int W_out = W_in/pool_size;
	int W_grid = ceilf((float)W_out/TILE_WIDTH);
	if(W_grid==0)
		W_grid = 1;
	n = blockIdx.x;
	m = blockIdx.y;
	h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y;
	w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x;

	//h and w is not center point of calculating, it's upper left corner point of Input image
	float acc = 0;
	for (p = 0; p < pool_size; p++) { // loop over KxK input samples
		for (q = 0; q < pool_size; q++)
			if(h < H_out && w < W_out)
				X[n*(M*H_in*W_in)+ m*(H_in*W_in) + (pool_size * h + p)*(W_in) + (pool_size * w + q)] =
						Y[n*(M*H_out*W_out)+ m*(H_out*W_out) + h*(W_out) + w];
	}
	__syncthreads();

}
