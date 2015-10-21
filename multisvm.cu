
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <cutil.h>
#include "cublas.h"
#include "src/common/cuTimer.cu"
#include "src/common/parseinputs.cpp"
#include "src/training/training.cu"
#include "src/testing/testing.cu"

void runmulticlassifier(	char* ,int ,int ,char* ,int ,int ,int ,int ,float ,float ,int ,float);

//MultiClass classification using SVM
int main(int argc, char** argv)
{

	CUT_DEVICE_INIT(argc, argv);

	runmulticlassifier(		    "/home/sherrero/NVIDIA_GPU_Computing_SDK/C/src/multisvm_1.0/data/adult/a9a",
								32561,
								123,
								"/home/sherrero/NVIDIA_GPU_Computing_SDK/C/src/multisvm_1.0/data/adult/a9a.t",
								16281,
								1,
								2,
								1,
								100,
								0.001,
								0,
								0.5);


	CUT_EXIT(argc, argv);
}

/**
 * Runs both training and testing. Provides timings
 * @param trainfilename name of the file containing the training samples
 * @param ntraining number of training samples
 * @param nfeatures number of features in the each training sample
 * @param testfilename name of the file containing the testing samples
 * @param ntesting number of testing samples
 * @param code {0: One vs All, 1: All vs All, 2: Even vs Odd}
 * @param nclasses number of classes in the SVM problem
 * @param ntasks number of binary classification tasks
 * @param C penalization parameter
 * @param tau stopping parameter of the SMO algorithm
 * @param kernelcode type of kernel to use
 * @param beta if using RBF kernel, the value of beta
 */

void runmulticlassifier(char* trainfilename,
						int ntraining,
						int nfeatures,
						char* testfilename,
						int ntesting,
						int code,
						int nclasses,
						int ntasks,
						float C,
						float tau,
						int kernelcode,
						float beta)
{

	cublasStatus status;

	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! CUBLAS initialization error\n");
	}

	float* h_C = (float*) malloc(sizeof(float) * ntasks);
	for(int i=0; i<ntasks; i++)
	{
		h_C[i]=C;
	}

	printf("Input Train File Name: %s\n", trainfilename);
	printf("Input Test File Name: %s\n", testfilename);

	if( code==0)
	{
		printf("Code: One Vs All\n");
	}
	else if( code==1)
	{
		printf("Code: All Vs All\n");
	}
	else if( code==2)
	{
		printf("Code: Even Vs Odd\n");
	}

	printf("# of training samples: %i\n", ntraining);
	printf("# of testing samples: %i\n", ntesting);
	printf("# of features: %i\n", nfeatures);
	printf("# of tasks: %i\n", ntasks);
	printf("# of classes: %i\n", nclasses);
	printf("Beta: %f\n", beta);

	//Allocate memory for xtraindata
	float* h_xtraindata = (float*) malloc(sizeof(float) * ntraining* nfeatures);
	float* h_xtraindatatemp = (float*) malloc(sizeof(float) * ntraining* nfeatures);

	//Allocate memory for xtestdata
	float* h_xtestdata = (float*) malloc(sizeof(float) * ntesting* nfeatures);
	float* h_xtestdatatemp = (float*) malloc(sizeof(float) * ntesting* nfeatures);


	//Allocate memory for ltraindata
	int* h_ltraindata = (int*) malloc(sizeof(int) * ntraining);

	//Allocate memory for ltestdata
	int* h_ltestdata = (int*) malloc(sizeof(int) * ntesting);


	//Parse data from input file
	printf("Parsing input data...\n");
	parsedatalibsvm(trainfilename, h_xtraindatatemp, h_ltraindata, ntraining, nfeatures, nclasses);
	parsedatalibsvm(testfilename, h_xtestdatatemp, h_ltestdata, ntesting, nfeatures, nclasses);
	printf("Parsing input data done!\n");

	//Allocate memory for rdata
	int* h_rdata= (int*) malloc(sizeof(int) * nclasses * ntasks);

	if( code==0)
	{
		generateovacode(h_rdata, nclasses, ntasks);
	}
	else if( code==1)
	{
		generateavacode(h_rdata, nclasses, ntasks);
	}
	else if(code==2)
	{
		generateevenoddcode(h_rdata, nclasses, ntasks);
	}

	printcode(h_rdata, nclasses, ntasks);

	bool iszero=false;

	for (int i=0; i< ntraining; i++)
	{
		for (int j=0; j<nfeatures; j++)
		{
			h_xtraindata[j*ntraining+i]=h_xtraindatatemp[i*nfeatures+j];
		}
		if(h_ltraindata[i]==0)
		{
			iszero=true;
		}
	}

	for (int i=0; i< ntesting; i++)
	{
		for (int j=0; j<nfeatures; j++)
		{
			h_xtestdata[j*ntesting+i]=h_xtestdatatemp[i*nfeatures+j];
		}
	}

	if (iszero)
	{
		for (int i=0; i< ntraining; i++)
		{
			h_ltraindata[i]=h_ltraindata[i]+1;
		}
		for (int i=0; i< ntesting; i++)
		{
			h_ltestdata[i]=h_ltestdata[i]+1;
		}
	}

	free(h_xtraindatatemp);
	free(h_xtestdatatemp);

	int* h_ltesthatdata = (int*) malloc(sizeof(int) * ntesting);

	//Allocate memory for b
	float * h_b= (float*) malloc(sizeof(float) * ntasks);
	for (int i=0; i<ntasks; i++)
	{
		h_b[i]= 0.0f;
	}

	//Allocate memory for adata
	float* h_atraindata= (float*) malloc(sizeof(int) * ntraining * ntasks);

	cuResetTimer();
	float tA1=cuGetTimer();
	printf("Training classifier...\n");
	trainclassifier			( 	h_xtraindata,
								h_ltraindata,
								h_rdata,
								h_atraindata,
								ntraining,
								nfeatures,
								nclasses,
								ntasks,
								h_C,
								h_b,
								tau,
								kernelcode,
								beta);

	float tA2=cuGetTimer();

	printf("Training classifier done!\n");
	printf("Training time Launch =%.1f usec, finished=%.1f msec\n",tA1*1.e3,tA2);

	for (int j=0; j<ntasks; j++)
	{
		int svnum=0;
		for (int i=0; i<ntraining; i++)
		{
			if(h_atraindata[j*ntraining + i]!=0)
			{
				svnum++;
			}
		}
		printf("Task %i, svnum, %i, b %f\n",j, svnum,h_b[j] );
	}

	int nSV=0;
	for (int i=0; i< ntraining; i++)
	{
		for (int j=0; j< ntasks; j++)
		{
			if(h_atraindata[j*ntraining+i]!=0)
			{
				nSV++;
				break;
			}
		}
	}

	float* h_xtraindatared = (float*) malloc(sizeof(float) * nSV* nfeatures);
	int* h_ltraindatared = (int*) malloc(sizeof(int) * nSV);
	float* h_atraindatared = (float*) malloc(sizeof(float) *ntasks* nSV);

	int k=0;

	for (int i=0; i< ntraining; i++)
	{
		//Check if SV
		bool isSV=false;

		for (int j=0; j< ntasks; j++)
		{
			if(h_atraindata[j*ntraining+i]!=0)
			{
				isSV=true;
				break;
			}
		}

		//If SV copy sample and alphas
		if(isSV)
		{
			for (int j=0; j< ntasks; j++)
			{
				h_atraindatared[j*nSV +k]= h_atraindata[j*ntraining+i];
			}


			for (int j=0; j<nfeatures; j++)
			{
				h_xtraindatared[j*nSV+k]=h_xtraindata[j*ntraining+i];
			}
			h_ltraindatared[k]= h_ltraindata[i];

			k++;
		}
	}


	printf("Testing classifier...\n");

	cuResetTimer();
	float tB1=cuGetTimer();
	testingclassifier(		h_xtraindatared,
							h_xtestdata,
							h_ltraindatared,
							h_ltesthatdata,
							h_rdata,
							h_atraindatared,
							nSV,
							ntesting,
							nfeatures,
							nclasses,
							ntasks,
							h_b,
							beta,
							kernelcode);

	printf("Testing classifier done\n");
	float tB2=cuGetTimer();
	printf("Testing time Launch =%.1f usec, finished=%.1f msec\n",tB1*1.e3,tB2);

	int errors=0;

	for (int i=0; i<ntesting; i++)
	{
		if( h_ltestdata[i]!=h_ltesthatdata[i])
		{
			errors++;
		}
	}


	printf("# of testing samples %i, # errors %i, Rate %f\n", ntesting, errors, 100* (float) (ntesting -errors)/(float)ntesting);

	free(h_rdata);
	free(h_xtraindata);
	free(h_xtestdata);
	free(h_ltraindata);
	free(h_ltestdata);
	free(h_b);
	free(h_atraindata);
	free(h_xtraindatared);
	free(h_ltraindatared);
	free(h_atraindatared);

}





