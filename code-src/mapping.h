/**
 * @file    mapping.h
 * @author  Saurabh jha <saurabh.jha.2010@gmail.com>
 * @brief  Mapping header file for threads, defines get_cpu_id method.
 * 
 *
 * (c) 2014, NTU Singapore, Xtra Group
 *
 */
#ifndef MAPPING_H
#define MAPPING_H

/** 
 * if the custom cpu mapping file exists, logical to physical mappings are
 * initialized from that file, otherwise it will be round-robin 
 */
#ifndef CUSTOM_CPU_MAPPING
#define CUSTOM_CPU_MAPPING "cpu-mapping.txt"
#endif

/**
 * Returns SMT aware logical to physical CPU mapping for a given thread id.
 */
int get_cpu_id(int thread_id,int n);



#endif 
