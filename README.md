# multisvm

The scaling of serial algorithms cannot rely on the improvement of CPUs anymore. The performance of classical Support Vector Machine (SVM) implementations has reached its limit and the arrival of the multi core era requires these algorithms to adapt to a new parallel scenario. Graphics Processing Units (GPU) have arisen as high performance platforms to implement data parallel algorithms. In this project, it is described how a naïve implementation of a multiclass classifier based on SVMs can map its inherent degrees of parallelism to the GPU programming model and efficiently use its computational throughput. Empirical results show that the training and classification time of the algorithm can be reduced an order of magnitude compared to a classical solver, LIBSVM, while guaranteeing the same accuracy.

We would appreciate feedback and any issues encountered.

These are some of the speedups reported during the training phase of the classifier:

| Dataset | Tasks     |	GPU (sec)  | LIBSVM (sec) | Speedup  |
|:-------:|:---------:|-----------:|-------------:|---------:|
| Adult   | Binary(2) |	32.67 	   | 341.5 	  | 10.45x   |
| Web 	  | Binary(2) |	156.95 	   | 2350.0 	  | 14.97x   |
| Mnist	  | Binary(2) |	425.89 	   | 13963.4 	  | 32.79x   |
| Usps 	  | Binary(2) | 1.65	   | 27.0	  | 16.36x   |
| Mnist	  | OVA(10)   |	2067.24	   | 118916.2	  | 57.52x   |
| Usps	  | OVA(10)   |	1.28	   | 21.3	  | 16.64x   |
| Shuttle | OVA(7)    |	5.85	   | 18.8	  | 3.38x    |
| Letter  | OVA(26)   |	19.04	   | 479.9	  | 25.20x   |

These are some of the speedups reported during the testing phase of the classifier:

|Dataset  | Tasks     | GPU (sec)  | LIBSVM (sec) | Speedup  |
|:-------:|:---------:|-----------:|-------------:|---------:|
| Adult	  | Binary(2) |	1.10	   | 42.7.0	  | 38.77x   |
| Web	  | Binary(2) |	2.51	   | 75.00	  | 29.88x   |
| Mnist	  | Binary(2) |	4.43	   | 496.50       | 112.19x  |
| Usps	  | Binary(2) |	0.07	   | 1.00 	  | 13.72x   |
| Mnist   | OVA(10)   |	14.00	   | 683.90	  | 48.85x   |
| Usps	  | OVA(10)   |	0.13	   | 3.62	  | 27.84x   |
| Shuttle | OVA(7)    |	0.49	   | 1.43	  | 2.92x    |
| Letter  | OVA(26)   |	2.02	   | 6.77	  | 3.35x    |