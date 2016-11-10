// Distributed two-dimensional Discrete FFT transform
// Jai Chauhan
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform1D(Complex* h, int N, Complex* H, int rn)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  for(int n = 0; n < N; n++)
    {
      for(int k = 0; k < N; k++)
	{
	  H[n + N*rn] = H[n + N*rn] + Complex(cos(2.0 * M_PI * n * k/N),-sin(2.0 * M_PI * n *k/N)) * h[k + N*rn];
	}
      if(fabs(H[n].imag) < 1e-10)
	{
	  H[n].imag = 0;
	}
      if(fabs(H[n].real) < 1e-10)
	{
	  H[n].real = 0;
	}
    }
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  int num,rc;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Number of LPs %d My rank = %d\n", num,rank);
  int w;
  int h ;
  w = image.GetWidth();
  h = image.GetHeight();
  Complex transform[w * h];
  int rows_each = h / num;
  Complex* data = image.GetImageData();
  int myrows = rank * w * rows_each; //start of my rows
  int numelements = w * rows_each;
  Complex mydata [numelements];
  Complex rowtransform [numelements];
  //send arrays
  double realvals[numelements];
  double imagvals[numelements];
  //request array
  MPI_Request realreq [num-1];
  MPI_Request imagreq [num-1];
  //to be filled arrays in rank 0
  double vals[num][numelements];
  double vals2[num][numelements];
  //transpose variables
  int newh = w;
  int neww = h;
  int rows_each2 = newh / num;
  int myrows2 = rank * neww * rows_each2; //start of my rows
  int numelements2 = neww * rows_each2;
  //2d request arrays
  MPI_Request realreq2 [num-1];
  MPI_Request imagreq2 [num-1];
  //send arrays
  double realvals2[numelements2];
  double imagvals2[numelements2];
  Complex mydata2 [numelements2];
  Complex rowtransform2 [numelements2];
  //transpose arrays
  Complex transpose[h][w];
  Complex linear_transpose[h * w];
  //double valsT[num][numelements2];
  //double vals2T[num][numelements2];
  //to be filled in rank0 2d
  double myarr[num][numelements2];
  double myarr2[num][numelements2];
  //done variables
  int done = 1;
  int done2 = 1;
  Complex final[h*w];
  //Begin 1D 
  if(rank == 0)
    {
      for(int r = 1; r < num; r++) 
	{
	  printf("Setting up recieves\n");
	  rc = MPI_Irecv(vals[r],numelements, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &realreq[r-1]);
	  //different tags for real vs imag
	  rc = MPI_Irecv(vals2[r],numelements, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &imagreq[r-1]);
	}
    }
  for(int i = 0; i < numelements; i++) 
  {
    //copy required rows from data into mydata
    mydata[i] = data[myrows + i];
  }
  for(int i = 0; i < rows_each; i++)
    {
      Transform1D(mydata, w, rowtransform, i);
    }
  printf("Back from transform\n");
  //seperate transform values  into Real and Imag
  for(int i = 0; i < numelements; i++)
    { 
      realvals[i] = rowtransform[i].real;
      imagvals[i] = rowtransform[i].imag;
    }
  if(rank != 0) 
    {
      MPI_Status status;
      MPI_Status status2;
      rc = MPI_Send(realvals,numelements,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
      rc = MPI_Send(imagvals,numelements,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
      //blocking recievie, wait for zero to signal completion
      rc = MPI_Recv(linear_transpose,w*h*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&status2);
      rc = MPI_Recv(&done,1,MPI_INT,0,0,MPI_COMM_WORLD, &status);
    }
  else 
    {
      for(int i = 0; i < numelements; i++)
	{
	  //transform[myrows + i] = Complex(realvals[i],imagvals[i]);
	  vals[0][i] = realvals[i];
	  vals2[0][i] = imagvals[i]; 
	}
      int count = 0;
      int count2 = 0;
      int index;
      MPI_Status status;
      MPI_Status imagstatus;  
      while(count <  15 || count2 < 15) 
	{ 
	  //printf("In while loop, the count is + %d\n",count);
	  if(count < 15)
	    {
	      rc = MPI_Waitany(num-1,realreq,&index,&status);
	      int source = status.MPI_SOURCE;
	      printf("Recieved reals from %d complete\n", source);
	      count ++;
	    }
	  if(count2 < 15)
	    {
	      rc = MPI_Waitany(num-1,imagreq,&index,&imagstatus);
	      int source = imagstatus.MPI_SOURCE;
	      printf("Recieved imags from %d complete\n", source);
	      count2++;
	      
	    }
	}

      //test 1d accuracy

      for(int i = 0; i < num; i++) 
	{
	  for(int j = 0; j < numelements; j++)
	    {
	      double real = vals[i][j];
	      double imag = vals2[i][j];
	      transform[j + numelements*i] = Complex(real, imag);
	    }
	}
      image.SaveImageData("MyAfter1D.txt", transform,w,h);
      
      //transpose
      //copy transform to square array
      Complex square_arr[h][w];
      for(int i = 0; i < h; i++)
	{
	  for(int j = 0; j < w; j++)
	    {
	      square_arr[i][j] = transform[w*i + j];
	    }
	}
      //transpose square_arr
      for(int i = 0; i<h; i++ )
	{
	  for(int j= 0; j < w; j++)
	    {
	      transpose[j][i] = square_arr[i][j];
	    }
	}
      //linearize tranpose
      for(int i = 0; i<newh; i++)
	{
	  for(int j = 0; j<neww; j++)
	    {
	      double vals = transpose[i][j].real;
	      if(vals == 0) 
		{
		  printf("A zero in tranpose!");
		}
		
	      //printf("The value in tranpose is %f\n", vals);
	      linear_transpose[(256*i) + j] = transpose[i][j];
	      double vals2 = linear_transpose[(256*i) + j].real;
		if(vals2 == 0)
		  {
		    printf("WTF is linear transpose is zero");
		  }
	    }
	}
      //send linear_tranpose to other ranks
      for(int i = 1; i < num; i++)
	{
	  rc = MPI_Send(linear_transpose,w*h*sizeof(Complex),MPI_CHAR,i,0,MPI_COMM_WORLD);
	} 
      for(int i = 1; i < num; i++)
	{
	  rc = MPI_Send(&done,1,MPI_INT,i,0,MPI_COMM_WORLD);
	}
    } //end of else i.e. rank zero only block
  if(rank == 0)
    {
      //initialize requests 2d
      for(int r = 1; r < num; r++) 
	{
	  printf("Setting up recieves 2nd time\n");
	  rc = MPI_Irecv(myarr[r],numelements2, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &realreq2[r-1]);
	  rc = MPI_Irecv(myarr2[r],numelements2, MPI_DOUBLE, r, 1, MPI_COMM_WORLD, &imagreq2[r-1]);
	  //rc = MPI_Recv(linear_transpose,w*h*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&status2);
	}
      
    }
  for(int i = 0; i < numelements2; i++) 
    {
      int num = (rank * 4096) + i;
      double value = linear_transpose[num].real;
      //printf("Starting index is %d\n", num);
      //printf("linear transpose here is  %f\n",value);
      mydata2[i] = linear_transpose[num];
    }
  for(int i = 0; i < rows_each2; i++)
    {
      Transform1D(mydata2, neww, rowtransform2, i);
    }
  printf("Back from transform2\n");
  //seperate transform values  into Real and Imag
  for(int i = 0; i < numelements2; i++)
    { 
      realvals2[i] = rowtransform2[i].real;
      imagvals2[i] = rowtransform2[i].imag;
    }
  if(rank != 0) 
      {
	MPI_Status status;
	rc = MPI_Send(realvals2,numelements2,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
	rc = MPI_Send(imagvals2,numelements2,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
	//blocking recievie, wait for zero to signal completion
	rc = MPI_Recv(&done2,1,MPI_INT,0,10,MPI_COMM_WORLD, &status);
      }
    else 
      {
	for(int i = 0; i < numelements2; i++)
	  {
	    myarr[0][i] = realvals2[i];
	    myarr2[0][i] = imagvals2[i]; 
	  }
      int count = 0;
      int count2 = 0;
      int index;
      MPI_Status status;
      MPI_Status imagstatus;
      /*
      for(int i = 0; i<num-1; i++) 
	{
	  rc = MPI_Wait(&realreq2[i],&status);
	  rc = MPI_Wait(&imagreq2[i],&status);
	  int source = status.MPI_SOURCE;
	  printf("2D:Recieved imags from %d complete\n", source);
	}
      */
      while(count <  15 || count2 < 15) 
	{ 
	  //printf("In while loop, the count is + %d\n",count);
	  if(count < 15)
	    {
	      rc = MPI_Waitany(num-1,realreq2,&index,&status);
	      int source = status.MPI_SOURCE;
	      printf("2D:Recieved reals from %d complete\n", source);
	      count ++;
	    }
	  if(count2 < 15)
	    {
	      rc = MPI_Waitany(num-1,imagreq2,&index,&imagstatus);
	      int source = imagstatus.MPI_SOURCE;
	      printf("2D:Recieved imags from %d complete\n", source);
	      count2++;
	      
	    }
	}

      //Combine myarr and myarr2
      for(int i = 0; i < num; i++) 
	{
	  for(int j = 0; j < numelements2; j++)
	    {
	      double real = myarr[i][j];
	      double imag = myarr2[i][j];
	      transform[(numelements2*i) + j] = Complex(real, imag);
	    }
	}
      //copy to square array
      Complex square_arr[newh][neww];
      for(int i = 0; i < newh; i++)
	{
	  for(int j = 0; j < neww; j++)
	    {
	      square_arr[i][j] = transform[neww*i + j];
	    }
	}
      //Going back to original:transpose square_arr
      for(int i = 0; i<newh; i++ )
	{
	  for(int j= 0; j < neww; j++)
	    {
	      transpose[j][i] = square_arr[i][j];
	    }
	}
      //linearize transpose
     for(int i = 0; i<h; i++)
	{
	  for(int j = 0; j<w; j++)
	    {
	      final[(w*i) + j] = transpose[i][j];
	    }
	}
     printf("SAVING FINAL\n");
     image.SaveImageData("MyAfter2D.txt",final,w,h);
      for(int i = 1; i < num; i++)
	{
	  rc = MPI_Send(&done2,1,MPI_INT,i,10,MPI_COMM_WORLD);
	}
      }//end of 2nd else
}




int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  int rc;
  rc = MPI_Init(&argc,&argv);
  if(rc != MPI_SUCCESS)
    {
      printf("Error starting MPI program.\n");
      MPI_Abort(MPI_COMM_WORLD,rc);
    }
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
  MPI_Finalize();
}  
  

  
