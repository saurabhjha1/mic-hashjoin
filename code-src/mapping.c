/**
 * @file    mapping.c
 * @author  Saurabh jha <saurabh.jha.2010@gmail.com>
 * 
 *  
 * @brief  Chooses between various thread affinity models
 * 
 *
 * (c) 2014, NTU Singapore, Xtra Group
 *
 */

#ifdef SCATTER
#include "scatter_threads.h"
#elif COMPACT
#include "compact_threads.h"
#else
#include "balanced_threads.h"
#endif
