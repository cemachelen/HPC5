/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>


/* Forward Declarations of utility functions*/
double get_current_time();
void datread(char*, void*, int, int);
void pgmwrite(char*, void*, int, int);
void checkCUDAError(const char*);


/* Dimensions of image */
#define WIDTH 256
#define HEIGHT 192

/* Number of iterations to run */
#define ITERATIONS 10

/* Dimensions of CUDA grid and block */
#define GRID_W  16
#define GRID_H  12
#define BLOCK_W 16
#define BLOCK_H 16

/* Maximum difference allowed between host result and GPU result */
#define MAX_DIFF 0.01

/* Data buffer to read edge data into */
float edge[HEIGHT][WIDTH];

/* Data buffer for the resulting image */
float img[HEIGHT][WIDTH];

/* Work buffers, with halos */
float host_input[HEIGHT+2][WIDTH+2];
float gpu_output[HEIGHT+2][WIDTH+2];
float host_output[HEIGHT+2][WIDTH+2];

/* The actual CUDA kernel that runs on the GPU - 1D version by column */
__global__ void inverseEdgeDetect1D_col(float *d_output, float *d_input, \
					float *d_edge)
{
  int col, row;
  int idx, idx_south, idx_north, idx_west, idx_east;
  int numcols = WIDTH + 2;

  /*
   * calculate global row index for this thread  
   * from blockIdx.x, blockDim.x and threadIdx.x
   * remember to add 1 to account for halo    
   */
  row = blockIdx.x*blockDim.x + threadIdx.x + 1;

  /*
   * loop over all columns of the image
   */
  for (col = 1; col <= WIDTH; col++) {
      /*
       * calculate linear index from col and row, for the centre
       * and neighbouring points needed below.
       * For the neighbouring points you need to add/subtract 1  
       * to/from the row or col indices.
       */
      
      idx = row * numcols + col;
      idx_south = (row - 1) * numcols + col;
      idx_north = (row + 1) * numcols + col;
      
      idx_west = row * numcols + (col - 1);
      idx_east = row * numcols + (col + 1);
      

      /* perform stencil operation */  
      d_output[idx] = (d_input[idx_south] + d_input[idx_west] \
		       + d_input[idx_north] + d_input[idx_east] - \
		       d_edge[idx]) * 0.25;

    }
}



/* The actual CUDA kernel that runs on the GPU - 1D version by row */
__global__ void inverseEdgeDetect1D_row(float *d_output, float *d_input, \
					float *d_edge)
{
  int col, row;
  int idx, idx_south, idx_north, idx_west, idx_east;
  int numcols = WIDTH + 2;

  /*
   * calculate global column index for this thread  
   * from blockIdx.x,blockDim.x and threadIdx.x    
   * remember to add 1 to account for halo     
   */
  col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  /*
   * loop over all rows of the image
   */
  for (row = 1; row <= HEIGHT; row++) {
      /*
       * calculate linear index from col and row, for the centre
       * and neighbouring points needed below.
       * For the neighbouring points you need to add/subtract 1  
       * to/from the row or col indices.
       */      
      idx = row * numcols + col;
      idx_south = (row - 1) * numcols + col;
      idx_north = (row + 1) * numcols + col;
      
      idx_west = row * numcols + (col - 1);
      idx_east = row * numcols + (col + 1);

      /* perform stencil operation */  
      d_output[idx] = (d_input[idx_south] + d_input[idx_west] \
		       + d_input[idx_north] + \
		       d_input[idx_east] - d_edge[idx]) * 0.25;
    }
}


/* The actual CUDA kernel that runs on the GPU - 2D version */
__global__ void inverseEdgeDetect2D(float *d_output, float *d_input, \
				    float *d_edge)
{
  int col, row;
  int idx, idx_south, idx_north, idx_west, idx_east;
  int numcols = WIDTH + 2;

  /*
   * calculate global column index for this thread  
   * from blockIdx.x,blockDim.x and threadIdx.x    
   * remember to add 1 to account for halo     
   */
  col = blockIdx.x*blockDim.x + threadIdx.x + 1;

  /*
   * calculate global row index for this thread  
   * from blockIdx.y,blockDim.y and threadIdx.y
   * remember to add 1 to account for halo    
   */
  row = blockIdx.y*blockDim.y + threadIdx.y + 1;
    

  /*
   * calculate linear index from col and row, for the centre
   * and neighbouring points needed below.
   * For the neighbouring points you need to add/subtract 1  
   * to/from the row or col indices.
   */
  idx = row * numcols + col;
  idx_south = (row - 1) * numcols + col;
  idx_north = (row + 1) * numcols + col;

  idx_west = row * numcols + (col - 1);
  idx_east = row * numcols +(col + 1);

  /* perform stencil operation */  
  d_output[idx] = (d_input[idx_south] + d_input[idx_west] + d_input[idx_north]
              + d_input[idx_east] - d_edge[idx]) * 0.25;
}


/* The actual CUDA kernel that runs on the GPU - 2D shared memory version */
/* Compare this with above 2D version */
__global__ void inverseEdgeDetect2D_sm(float *d_output, float *d_input, \
				    float *d_edge)
{
  int col, row, col_sm, row_sm;
  int idx, idx_south_sm, idx_north_sm, idx_west_sm, idx_east_sm;
  int numcols = WIDTH + 2;
  int numcols_sm = BLOCK_W + 2; 

  /* shared memory array for holding block of input array */
  __shared__ float d_input_sm[(BLOCK_H+2)*(BLOCK_W+2)];

  /* col, row correspond to location in global array */
  col = blockIdx.x*BLOCK_W + threadIdx.x;
  row = blockIdx.y*BLOCK_H + threadIdx.y;

  /* col_sm, row_sm correspond to location in shared memory block */
  col_sm = threadIdx.x;    
  row_sm = threadIdx.y;    

  /* load block of d_input into shared memory, including halo regions */
  d_input_sm[row_sm*numcols_sm+col_sm] = d_input[row*numcols + col];
  
  /* synchronise to ensure that all elements are loaded */
  syncthreads();

  /* only threads corresponding to non-halo elements should now be active */ 
  if (row_sm > 0 && row_sm <= BLOCK_H && 
      col_sm > 0 && col_sm <= BLOCK_W )
    {
      
      /* calculate indices */
      idx = row * numcols + col;
      idx_south_sm = (row_sm - 1) * numcols_sm + col_sm;
      idx_north_sm = (row_sm + 1) * numcols_sm + col_sm;
      idx_west_sm = row_sm * numcols_sm +  (col_sm - 1);
      idx_east_sm = row_sm * numcols_sm +  (col_sm + 1);
            
      /* perform stencil operation */     
      d_output[idx] = (d_input_sm[idx_south_sm] + d_input_sm[idx_west_sm] + 
		       d_input_sm[idx_north_sm] + d_input_sm[idx_east_sm] 
		       - d_edge[idx]) * 0.25;
    }
}



int main(int argc, char *argv[])
{
  int x, y;
  int i;
  int errors;

  double start_time_inc_data, end_time_inc_data;
  double cpu_start_time, cpu_end_time;

  float *d_input, *d_output, *d_edge, *tmp;

  size_t memSize = (WIDTH+2) * (HEIGHT+2) * sizeof(float);

  printf("Grid size: %dx%d\n", GRID_W, GRID_H);
  printf("Block size: %dx%d\n", BLOCK_W, BLOCK_H);

  /* allocate memory on device */
  cudaMalloc((void**)&d_input, memSize);
  cudaMalloc((void**)&d_output, memSize);
  cudaMalloc((void**)&d_edge, memSize);

  /* read in edge data */
  datread("edge256x192.dat", (void *)edge, WIDTH, HEIGHT);

  /* zero buffer so that halo is zeroed */
  for (y = 0; y < HEIGHT+2; y++) {
    for (x = 0; x < WIDTH+2; x++) {
      host_input[y][x] = 0.0;
    }
  }

  /* copy input to buffer with halo */
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
       host_input[y+1][x+1] = edge[y][x];
    }
  }

  /*
   * copy to all the GPU arrays. d_output doesn't need to have this data but
   * this will zero its halo
   */
  start_time_inc_data = get_current_time();
  cudaMemcpy( d_input, (void *)host_input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_output, (void *)host_input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_edge, (void *)host_input, memSize, cudaMemcpyHostToDevice);

  /* run on GPU */
  for (i = 0; i < ITERATIONS; i++) {
    /* run the kernel */
    /*
     * One of these kernel invocations should be uncommented at a time
     */


    //dim3 dimGrid(GRID_H);
    //dim3 dimBlock(BLOCK_H);
    //inverseEdgeDetect1D_col<<< dimGrid, dimBlock >>>(d_output, d_input, d_edge);

    //dim3 dimGrid(GRID_W);
    //dim3 dimBlock(BLOCK_W);
    //inverseEdgeDetect1D_row<<< dimGrid, dimBlock >>>(d_output, d_input, d_edge);

    //dim3 dimGrid(GRID_W, GRID_H);
    //dim3 dimBlock(BLOCK_W, BLOCK_H);
    //inverseEdgeDetect2D<<< dimGrid, dimBlock >>>(d_output, d_input, d_edge);
 
    dim3 dimGrid(GRID_W, GRID_H);
    dim3 dimBlock(BLOCK_W+2, BLOCK_H+2);
    inverseEdgeDetect2D_sm<<< dimGrid, dimBlock >>>(d_output, d_input, d_edge);




    cudaThreadSynchronize();

    /* swap the buffer pointers ready for next time */
    tmp = d_input;
    d_input = d_output;
    d_output = tmp;
  }

  cudaMemcpy((void *)gpu_output, d_input, memSize, cudaMemcpyDeviceToHost);

  end_time_inc_data = get_current_time();

  /*
   * run on host for comparison
   */
  cpu_start_time = get_current_time();
  for (i = 0; i < ITERATIONS; i++) {

    /* perform stencil operation */
    for (y = 0; y < HEIGHT; y++) {
      for (x = 0; x < WIDTH; x++) {
	host_output[y+1][x+1] = (host_input[y+1][x] + host_input[y+1][x+2] +
				 host_input[y][x+1] + host_input[y+2][x+1] \
				 - edge[y][x]) * 0.25;
      }
    }
    
    /* copy output back to input buffer */
    for (y = 0; y < HEIGHT; y++) {
      for (x = 0; x < WIDTH; x++) {
	host_input[y+1][x+1] = host_output[y+1][x+1];
      }
    }
  }
  cpu_end_time = get_current_time();

  /* check that GPU result matches host result */
  errors = 0;
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      float diff = fabs(gpu_output[y+1][x+1] - host_output[y+1][x+1]);
      if (diff >= MAX_DIFF) {
        errors++;
        printf("Error at %d,%d (CPU=%f, GPU=%f)\n", x, y, \
	       host_output[y+1][x+1],\
               gpu_output[y+1][x+1]);
      }
    }
  }
  if (errors == 0) printf("Correct\n");

  /* copy result to output buffer */
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      img[y][x] = gpu_output[y+1][x+1];
    }
  }

  /* write PGM */
  pgmwrite("output.pgm", (void *)img, WIDTH, HEIGHT);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_edge);

  printf("GPU Time (Including Data Transfer): %fs\n", \
	 end_time_inc_data - start_time_inc_data);
  printf("CPU Time                          : %fs\n", \
	 cpu_end_time - cpu_start_time);

  return 0;
}


/* Utility Functions */

/*
 * Function to get an accurate time reading
 */
double get_current_time()
{
   static int start = 0, startu = 0;
   struct timeval tval;
   double result;

   if (gettimeofday(&tval, NULL) == -1)
      result = -1.0;
   else if(!start) {
      start = tval.tv_sec;
      startu = tval.tv_usec;
      result = 0.0;
   }
   else
      result = (double) (tval.tv_sec - start) + 1.0e-6*(tval.tv_usec - startu);

   return result;
}


/* Read the input file containing the edge data */
void datread(char *filename, void *vx, int nx, int ny)
{ 
  FILE *fp;

  int nxt, nyt, i, j, t;

  float *x = (float *) vx;

  if (NULL == (fp = fopen(filename,"r")))
  {
    fprintf(stderr, "datread: cannot open <%s>\n", filename);
    exit(-1);
  }

  fscanf(fp,"%d %d",&nxt,&nyt);

  if (nx != nxt || ny != nyt)
  {
    fprintf(stderr,
            "datread: size mismatch, (nx,ny) = (%d,%d) expected (%d,%d)\n",
            nxt, nyt, nx, ny);
    exit(-1);
  }

  for (j=0; j<ny; j++)
  {
    for (i=0; i<nx; i++)
    {
      fscanf(fp,"%d", &t);
      x[(ny-j-1)*nx + i] = t;
    }
  }

  fclose(fp);
}

/* Write the output image as a PGM file */
void pgmwrite(char *filename, void *vx, int nx, int ny)
{
  FILE *fp;

  int i, j, k, grey;

  float xmin, xmax, tmp;
  float thresh = 255.0;

  float *x = (float *) vx;

  if (NULL == (fp = fopen(filename,"w")))
  {
    fprintf(stderr, "pgmwrite: cannot create <%s>\n", filename);
    exit(-1);
  }

  /*
   *  Find the max and min absolute values of the array
   */

  xmin = fabs(x[0]);
  xmax = fabs(x[0]);

  for (i=0; i < nx*ny; i++)
  {
    if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
    if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
  }

  fprintf(fp, "P2\n");
  fprintf(fp, "# Written by pgmwrite\n");
  fprintf(fp, "%d %d\n", nx, ny);
  fprintf(fp, "%d\n", (int) thresh);

  k = 0;

  for (j=ny-1; j >=0 ; j--)
  {
    for (i=0; i < nx; i++)
    {
      /*
       *  Access the value of x[i][j]
       */

      tmp = x[j*nx+i];

      /*
       *  Scale the value appropriately so it lies between 0 and thresh
       */

      if (xmin < 0 || xmax > thresh)
      {
        tmp = (int) ((thresh*((fabs(tmp-xmin))/(xmax-xmin))) + 0.5);
      }
      else
      {
        tmp = (int) (fabs(tmp) + 0.5);
      }

      /*
       *  Increase the contrast by boosting the lower values
       */
     
      grey = (int) (thresh * sqrt(tmp/thresh));

      fprintf(fp, "%3d ", grey);

      if (0 == (k+1)%16) fprintf(fp, "\n");

      k++;
    }
  }

  if (0 != k%16) fprintf(fp, "\n");
  fclose(fp);
}

/* Simple utility function to check for CUDA runtime errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

