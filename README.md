mic-hashjoin
============


Authors
-------

	* Saurabh Jha - NTU Singapore
	* Mian LU - IHPC Singapore
	* Bingsheng He- NTU Singapore
	* Huynh Phung Huynh - IHPC Singapore
<br>
 (c) 2014, NTU Singapore, Xtra Group
 
==============================================================================================================

Hash join implementation for Xeon Phi (mic) and CPU's
===============================================================================================================
Compilation Steps on MIC Cards
==============================
1. cd code-src
2. icc *.c -pthread -std=c99 -O3 -mmic -DPREFETCH_NPJ -DSIMD -DSWWC -DHUGE_PAGES -DPDSIT=64 -DSKEW_HANDLING

	* -DSIMD	---> enable SIMD Vectorization
	* -DPREFETCH_NPJ---> enable manual prefetching in NPO, mNPO
	* -DSWWC 	---> enable software managed buffers
	* -DHUGE_PAGES	---> enable huge pages
	* -DPDIST 	---> enable setting a manual prefetch distance for mPRO,mPRH,mPRHO 
	* -DSKEW_HANDLING ---> enable skew handling [enabled by default]
	* -DKEY_8B	---> enable 64bit workload [not shown, add to list of arguments to enable]
	* -DSYNCSTATS	---> enable to show synchronization statistics [not shown, add to list of arguments to enable]
	* -DCOMPACT	---> enable compact thread affinity
	* -DBALANCED	---> enable balance thread affinity [default]
	* -DSCATTER	---> enable scatter thread affinity

Running code on MIC Cards
=========================
1. Initialize runOnmic.sh script with proper environment variables. 
	HOST is mic card to use, generally number as mic0, mic1, mic2, mic3
	USER is the user name of the user having access to mic cards. It is advisible to setup passwordless ssh.
2. ./runOnmic.sh ./a.out "-a PRO -n 180"
	
	Join algorithm selection, algorithms : PRO, PRH, PRHO, NPO
	* -a --algo=&lt; name &gt;    Run the hash join algorithm named &lt; name &gt; [PRO]
	On mic cards, PRO refers to mPRO, PRHO refers mPRHO, PRH refers to mPRH and NPO refers to mNPO
	Other join configuration options, with default values in [] 
	* -n --nthreads= &lt; N &gt;  Number of threads to use &lt; N &gt; [2]
	* -r --r-size=&lt; R &gt;    Number of tuples in build relation R &lt; R &gt; [128000000]
	* -s --s-size=&lt; S &gt;    Number of tuples in probe relation S &lt; S &gt; [128000000]
	* -x --r-seed=&lt; x &gt;    Seed value for generating relation R &lt; x &gt; [12345]
	* -y --s-seed=&lt; y &gt;    Seed value for generating relation S &lt; y &gt; [54321]
	* -z --skew=&lt; z &gt;      Zipf skew parameter for probe relation S &lt; z &gt; [0.0]
	* --non-unique       Use non-unique (duplicated) keys in input relations 
	* --full-range       Spread keys in relns. in full 32-bit integer range

Compilation on CPU
==================
1. cd code-src 
2. icc *.c -pthread -std=c99 -O3 -DSWWC -DCOMPACT
Note -DSCATTER and -DBALANCE not applicable for CPU and -DCOMPACT must be used.

Running code on CPU
===================
1. ./a.out -a PRO -n 16
2. Other options and arguments same as shown for mic cards.


==================================================================================================================
==================================================================================================================



