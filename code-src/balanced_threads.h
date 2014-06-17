/**
 * @file    balanced_threads.h
 * @author  Saurabh jha <saurabh.jha.2010@gmail.com>
 *  
 * @brief  balancer method provides an interface to assign threads on 
 * 	   to the physical cores in much the same way as the openmp
 *	   balance thread affinity. 
 * @Note   this method only applies to mic cards.
 *
 * (c) 2014, NTU Singapore, Xtra Group
 *
 */

#ifdef __MIC__
#include <stdio.h>  /* FILE, fopen */
#include <stdlib.h> /* exit, perror */
#include <unistd.h> /* sysconf */
#include <math.h>
#include "mapping.h"

// Define number of parameters supported on mic cards, future proofing the method.
#define MAX_NODES 512

#define MIC_CORES 60

#define MIC_HT 4

static int inited = 0;
static int max_cpus;
static int node_mapping[MAX_NODES];
static int cpu[MIC_CORES];
static int mfree=0;


/**
 * Returns SMT aware logical to physical CPU mapping for a given thread id.
 */
 

/* On mic platforms creating more than 240 threads cannot speed up the application because of its in order execution model,
   However, the below code takes care of any number of threads that needs to be spawned in a balanced way.
*/



int balancer(int thread_id, int nthreads)
{


	int tpc =( nthreads/MIC_CORES) ; //calculate the occupancy depending on cores and threads that needs to be created
	int tid;
	if (nthreads/MIC_CORES < 1) tpc = 1 ;
	if(cpu[mfree]<tpc)
	{
	   tid =((mfree*MIC_HT) + (cpu[mfree]%MIC_HT))%240;
	   cpu[mfree]++;
	   
	}
	else
	{
	   mfree++;
	   tid = ((mfree*MIC_HT) + (cpu[mfree]%MIC_HT))%(240);
	  
	   cpu[mfree]++;
	   
	}

return tid; 
} 

int 
get_cpu_id(int thread_id, int nthreads) 
{
return balancer(thread_id,nthreads);
}

#else 
#error "Balanced mode only for mic cards"
#endif
