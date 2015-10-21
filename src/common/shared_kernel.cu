#ifndef _SHARED_KERNEL_H_
#define _SHARED_KERNEL_H_

#include <stdio.h>

#define TPB 128
#define LOADX 32
#define LOADY 32

/**
 * Calculates the dot product of one sample with another
 * @param d_x1data input sample 1
 * @param d_x2data input sample 2
 * @param d_Ix1 index of sample 1
 * @param d_Ix2 index of smaple 2
 * @param d_noffset1 offset in the input array
 * @param d_noffset2 offset in the input array
 * @param nfeatures number of features of the input points
 */
__device__ inline float LinearKernel(float* d_x1data, float* d_x2data, int Ix1, int Ix2, int noffset1, int noffset2,  int nfeatures)
{
	float result=0;

	for (int i=0; i< nfeatures; i++)
	{
		result += d_x1data[Ix1 + i*noffset1] * d_x2data [Ix2 + i*noffset2];
	}
	return result;
}


/**
 * Calculates the RBF value
 * @param d_x1data input sample 1
 * @param d_x2data input sample 2
 * @param d_Ix1 index of sample 1
 * @param d_Ix2 index of smaple 2
 * @param d_noffset1 offset in the input array
 * @param d_noffset2 offset in the input array
 * @param nfeatures number of features of the input points
 * @param beta parameter of the RBF kernel
 */
__device__ inline float RbfKernel(float* d_x1data, float* d_x2data, int Ix1, int Ix2, int noffset1,int noffset2, int nfeatures,float beta)
{
	float result=0;

	for (int i=0; i< nfeatures; i++)
	{
		result += (d_x1data[Ix1 + i*noffset1] - d_x2data [Ix2 + i*noffset2])*  (d_x1data[Ix1 + i*noffset1] - d_x2data [Ix2 + i*noffset2]);
	}
	result= expf(-beta* result);

	return result;
}


/**
 * Calculates the RBF value
 * @param d_xdata input vector
 * @param d_kdata matrix containing rows of the gram matrix
 * @param Irow row of the input set that needs to be calculated
 * @param Icache position that the calculated row will occupy
 * @param ntraining number of training samples
 * @param nfeatures number of features in each sample
 * @param beta parameter of the RBF kernel
 * @param kernelcode type of kernel considered
 */

__global__  void Kernel( 			float* d_xdata,
									float* d_kdata,
									int Irow,
									int Icache,
									int ntraining,
									int nfeatures,
									float beta,
									int kernelcode)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int bidx = blockIdx.x;

	int blockX=(int) ceilf((float)(ntraining)/(float)(TPB));

	if((bidx != blockX-1) || ((bidx == blockX-1) && tid < (ntraining - bidx*TPB)))
	{
		int point= bidx*TPB +tid;

		if(kernelcode==0)
		{
			d_kdata[Icache*ntraining + point]=RbfKernel(d_xdata,d_xdata, Irow, point, ntraining,ntraining,  nfeatures,beta);
		}
		else if(kernelcode==1)
		{
			d_kdata[Icache*ntraining + point]= LinearKernel(d_xdata,d_xdata, Irow, point,ntraining, ntraining, nfeatures);
		}


	}

	__syncthreads();
}


/**
 * Extract the vector to be computed from the training set
 * @param d_xdata input training set
 * @param d_kernelrow will store the extracted point
 * @param Irow index of the vector of the training set that is considered
 * @param ntraining number of samples in the training set
 * @param nfeatures number of features in each training sample
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__  void ExtractKernelRow( 			float* d_xdata,
											float* d_kernelrow,
											int Irow,
											int ntraining,
											int nfeatures)
{

	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;


	while (i < nfeatures)
	{
		d_kernelrow[i]= d_xdata[ Irow + (i)* ntraining];

		if (isNtrainingPow2 || i + blockSize < ntraining)
		{

			d_kernelrow[i + blockSize]= d_xdata[ Irow + (i + blockSize)* ntraining];
		}

		i += gridSize;
	}

	__syncthreads();

}

/**
 * Set the result of the kernel evaluation in the cache matrix
 * @param d_kdata cache that will keep the result
 * @param d_dottraindata product of the input sample with itself
 * @param d_kernelrow input vector to be considered
 * @param Irow index of the d_dotraindata to be considered
 * @param Icache index of the row in the cache that the result will occupy
 * @param ntraining number of samples in the training set
 * @param beta parameter for the RBF kernel
 * @param kernelcode type of kernel to compute
 */

template <unsigned int blockSize, bool isNtrainingPow2>
__global__  void SetKernelDot(	 			float* d_kdata,
											float* d_dottraindata,
											float* d_kernelrow,
											int Irow,
											int Icache,
											int ntraining,
											float beta,
											int kernelcode)
{

	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;


	while (i < ntraining)
	{
		if(kernelcode==0)
		{
			float val= 2* beta* d_kernelrow [i] - beta* (d_dottraindata[i] + d_dottraindata[Irow]);
			d_kdata[Icache*ntraining + i] = expf(val);

			if (isNtrainingPow2 || i + blockSize < ntraining)
			{
				val= 2* beta* d_kernelrow [i + blockSize] - beta* (d_dottraindata[i + blockSize] + d_dottraindata[Irow]);
				d_kdata[Icache*ntraining + i + blockSize] = expf(val);
			}
		}
		else if (kernelcode==1)
		{
			d_kdata[Icache*ntraining + i]= d_kernelrow [i];
		}

		i += gridSize;
	}
	//}
	__syncthreads();


}


#endif // _SHARED_KERNEL_H_
