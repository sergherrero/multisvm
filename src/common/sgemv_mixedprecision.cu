

/*This mixed-precision matrix-vector multiplication algorithm is based on cublasSgemv NVIDIA's CUBLAS 1.1.
*/

#define LOG_THREAD_COUNT    (7)
#define THREAD_COUNT        (1 << LOG_THREAD_COUNT)
#define CTAS                (64)
#define IDXA(row,col)       (lda*(col)+(row))
#define IDXX(i)             (startx + ((i) * incx))
#define IDXY(i)             (starty + ((i) * incy))
#define TILEW_LOG           (5)
#define TILEW               (1 << TILEW_LOG)
#define TILEH_LOG           (5)
#define TILEH               (1 << TILEH_LOG)
#define X_ELEMS_PER_THREAD  (4)
#define IINC                (CTAS * THREAD_COUNT)
#define JINC                (THREAD_COUNT * X_ELEMS_PER_THREAD)
#define XINC                (THREAD_COUNT)


__shared__ float XX[TILEH];
__shared__ float AA[(TILEH+1)*TILEW];


__global__ void sgemvn_mixedprecis(const float *A, const float *x,float *y, int m, int n, int lda, int   incx,   int   incy)
{
    __shared__ float XX[JINC];
    int i, ii, j, jj, idx, incr, tid;
    double sdot;
    int startx;
    int starty;


    tid = threadIdx.x;
    startx = (incx >= 0) ? 0 : ((1 - n) * incx);
    starty = (incy >= 0) ? 0 : ((1 - m) * incy);

    for (i = 0; i < m; i += IINC) {

        ii = i + blockIdx.x * THREAD_COUNT;
        if (ii >= m) break;
        ii += tid;
        sdot = 0.0f;

        for (j = 0; j < n; j += JINC) {
            int jjLimit = min (j + JINC, n);
            incr = XINC * incx;
            jj = j + tid;
            __syncthreads ();
            idx = IDXX(jj);

            if (jj < (jjLimit - 3 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
                XX[tid+3*XINC] = x[idx + 3 * incr];
            }
            else if (jj < (jjLimit - 2 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
                XX[tid+2*XINC] = x[idx + 2 * incr];
            }
            else if (jj < (jjLimit - 1 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
                XX[tid+1*XINC] = x[idx + 1 * incr];
            }
            else if (jj < (jjLimit - 0 * XINC)) {
                XX[tid+0*XINC] = x[idx + 0 * incr];
            }

            __syncthreads ();

            if (ii < m) { /* if this row is active, accumulate dp */
                idx = IDXA(ii, j);
                incr = lda;
                jjLimit = jjLimit - j;
                jj = 0;
                while (jj < (jjLimit - 5)) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    sdot += A[idx + 1*incr] * XX[jj+ 1];
                    sdot += A[idx + 2*incr] * XX[jj+ 2];
                    sdot += A[idx + 3*incr] * XX[jj+ 3];
                    sdot += A[idx + 4*incr] * XX[jj+ 4];
                    sdot += A[idx + 5*incr] * XX[jj+ 5];
                    jj   += 6;
                    idx  += 6 * incr;
                }
                while (jj < jjLimit) {
                    sdot += A[idx + 0*incr] * XX[jj+ 0];
                    jj   += 1;
                    idx  += 1 * incr;
                }
            }
        }
        if (ii < m) {
            idx = IDXY(ii);

           y[idx] = sdot;
        }
    }
}
