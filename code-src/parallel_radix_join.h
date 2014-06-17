/**
 * @file    parallel_radix_join.h
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date    Sun Feb 20:09:59 2012
 * @version $Id: parallel_radix_join.h 3017 2012-12-07 10:56:20Z bcagri $
 * 
 * @brief  Provides interfaces for several variants of Radix Hash Join.
 * @note   File modified by Xtra group to add details for Xeon Phi
 * 
 * (c) 2012, ETH Zurich, Systems Group
 * 
 */

#ifndef PARALLEL_RADIX_JOIN_H
#define PARALLEL_RADIX_JOIN_H

#include "types.h" /* relation_t */

/** 
 * PRO: Parallel Radix Join Optimized.
 *
 * The "Parallel Radix Join Optimized" implementation denoted as PRO implements
 * the parallel radix join idea by Kim et al. with further optimizations. Mainly
 * it uses the bucket chaining for the build phase instead of the
 * histogram-based relation re-ordering and performs better than other
 * variations such as PRHO, which applies SIMD and prefetching
 * optimizations.
 
 * @note The same function will act as mPRO for mic cards.
 * @param relR  input relation R - inner relation 
 * @param relS  input relation S - inner relation 
 * 
 * @return number of result tuples
 */
int64_t 
PRO(relation_t * relR, relation_t * relS, int nthreads);

/** 
 * RJ: Radix Join.
 *
 * The "Radix Join" implementation denoted as RJ implements 
 * the single-threaded original multipass radix cluster join idea 
 * by Manegold et al. 
 *
 * * @note The same function will work as it is on mic cards.
 * @param relR  input relation R - inner relation
 * @param relS  input relation S - inner relation 
 * 
 * @warning nthreads parameter does not have any effect for this algorithm.
 * @return number of result tuples
 */
int64_t 
RJ(relation_t * relR, relation_t * relS, int nthreads);

/** 
 * PRH: Parallel Radix Join Histogram-based.
 *
 * The "Parallel Radix Join Histogram-based" implementation denoted as PRH
 * implements the parallel radix join idea by Kim et al. without SIMD and
 * prefetching optimizations. The difference from PRO is that the build phase
 * is based on the histogram-based relation re-ordering idea.

 * @note The same function will work as mPRH on mic cards.

 * @param relR  input relation R - inner relation 
 * @param relS  input relation S - inner relation 
 * 
 * @return number of result tuples
 */
int64_t 
PRH(relation_t * relR, relation_t * relS, int nthreads);

/** 
 * PRHO: Parallel Radix Join Histogram-based Optimized.
 *
 * The "Parallel Radix Join Histogram-based Optimized" implementation denoted 
 * as PRHO implements the parallel radix join idea by Kim et al. with SIMD and
 * prefetching optimizations. The difference from PRH is that the probe phase
 * uses SIMD and software prefetching optimizations.
 
 * @note The same function will act as mPRHO for mic cards. 64 bit not supported

 * @param relR  input relation R - inner relation
 * @param relS  input relation S - inner relation 
 * 
 * @return number of result tuples
 */
int64_t 
PRHO(relation_t * relR, relation_t * relS, int nthreads);

#endif /* PARALLEL_RADIX_JOIN_H */
