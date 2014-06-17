/**
 * @file    parallel_radix_join.c
 * @author  Saurabh jha <saurabh.jha.2010@gmail.com>
 * @brief  Chooses between cpu and mic
 * @Note   this method only applies to mic cards.
 *
 * (c) 2014, NTU Singapore, Xtra Group
 *
 */


#ifdef KEY_8B
#ifdef __MIC__
#include "mic_64.h"
#else
#include "cpu_hash.h"
#endif
#else
#ifdef __MIC__
#include "mic_hash.h"
#else
#include "cpu_hash.h"

#endif
#endif
