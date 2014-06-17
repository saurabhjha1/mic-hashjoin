/**
 * @file    scatter_threads.h
 * @author  Saurabh jha <saurabh.jha.2010@gmail.com>
 *  
 * @brief  scatter method provides an interface to assign threads on 
 * 	   to the physical cores in much the same way as the openmp
 *	   scatter thread affinity. 
 * @Note   this method only applies to mic cards.
 *
 * (c) 2014, NTU Singapore, Xtra Group
 *
 */
#include <stdio.h>  /* FILE, fopen */
#include <stdlib.h> /* exit, perror */
#include <unistd.h> /* sysconf */

#include "mapping.h"

/** \internal
 * @{
 */
#define USE_SOCKET 1
#define FRONT_NODES 30
#define BACK_NODES 30
#define MIC_HT 4 // threads per code
#define CORE 60
#define THREADS 240
static int inited = 0;
static int max_cpus;

static int cpu_used;
static int fcpu_used[FRONT_NODES] ={0};
static int lcpu_used[BACK_NODES] ={0};
static int contention = 1; //default contention is 1
static int flip=0; //0 means,1 means back
static int front=0,back=0;
static int tpc=4; //default set to 4



/* Trying to create a scatter scheduler */
int scatter(int thread_id,int nthreads)
{

if(!inited)
{
	tpc= nthreads/CORE;
}
int set;
if(flip == 0)
{
	flip=1;
	if(front<FRONT_NODES)
	{
		
		set = front*MIC_HT +fcpu_used[front];
		fcpu_used[front]++;
		if(fcpu_used[front]==tpc)
		front++;
	}
	//printf("\nfrontset:%d",set);
}
else{
	flip=0;
	if(back<BACK_NODES)
	{
		
		set = (CORE-back-1)*MIC_HT+lcpu_used[back];
		lcpu_used[back]++;
		if(lcpu_used[back]==tpc)
		back++;
	
	}
	//printf("\nbackset:%d",set);
}

return set;
}


int get_cpu_id(int thread_id,int num_threads)
{
return scatter(thread_id,num_threads);
}
