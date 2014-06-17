/**
 * @file    main.c
 * @author  Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @version $Id: main.c 3017 2012-12-07 10:56:20Z bcagri $
 *
 * @brief  Main entry point for running join implementations with given command
 * line parameters.
 *
 * (c) 2012, ETH Zurich, Systems Group
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>              /* sched_setaffinity */
#include <stdio.h>              /* printf */
#include <sys/time.h>           /* gettimeofday */
#include <getopt.h>             /* getopt */
#include <stdlib.h>             /* exit */
#include <string.h>             /* strcmp */
#include <limits.h>             /* INT_MAX */

#include "no_partitioning_join.h" /* no partitioning joins: NPO, NPO_st */
#include "parallel_radix_join.h"  /* parallel radix joins: RJ, PRO, PRH, PRHO */
#include "generator.h"            /* create_relation_xk */


#include "affinity.h"      /* pthread_attr_setaffinity_np & sched_setaffinity */
#include "../config.h"     /* autoconf header */
#include "prj_params.h"
#if !defined(__cplusplus)
int getopt(int argc, char * const argv[],
           const char *optstring);
#endif

typedef struct algo_t  algo_t;
typedef struct param_t param_t;

struct algo_t
{
    char name[128];
    int64_t (*joinAlgo)(relation_t * , relation_t *, int);
};

struct param_t
{
    algo_t * algo;
    uint32_t nthreads;
    uint32_t r_size;
    uint32_t s_size;
    uint32_t r_seed;
    uint32_t s_seed;
    double skew;
    int nonunique_keys;  /* non-unique keys allowed? */
    int verbose;
    int fullrange_keys;  /* keys covers full int range? */
    int basic_numa;/* alloc input chunks thread local? */
    char * perfconf;
    char * perfout;
};

extern char * optarg;
extern int    optind, opterr, optopt;

/** An experimental feature to allocate input relations numa-local */
extern int numalocalize;  /* defined in generator.c */
extern int nthreads;      /* defined in generator.c */

/** all available algorithms */
static struct algo_t algos [] =
{
    {"PRO", PRO},
    {"RJ", RJ},
    {"PRH", PRH},
    {"PRHO", PRHO},
    {"NPO", NPO},
    {"NPO_st", NPO_st}, /* NPO single threaded */
    {{0}, 0}
};

/* command line handling functions */
void
print_help();

void
print_version();

void
parse_args(int argc, char ** argv, param_t * cmd_params);

int
main(int argc, char ** argv)
{
    relation_t relR;
    relation_t relS;
    int64_t    results;

    /* start initially on CPU-0 */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    if (sched_setaffinity(0, sizeof(set), &set) <0)
    {
        perror("sched_setaffinity");
    }

    /* Command line parameters */
    param_t cmd_params;

    /* Default values if not specified on command line */
    cmd_params.algo     = &algos[0]; /* PRO */
    cmd_params.nthreads = 2;
    /* default dataset is Workload B (described in paper) */
    cmd_params.r_size   = 128000000;
    cmd_params.s_size   = 128000000;
    cmd_params.r_seed   = 12345;
    cmd_params.s_seed   = 54321;
    cmd_params.skew     = 0.0;
    cmd_params.verbose  = 0;
    cmd_params.perfconf = NULL;
    cmd_params.perfout  = NULL;
    cmd_params.nonunique_keys   = 0;
    cmd_params.fullrange_keys   = 0;
    cmd_params.basic_numa = 0;

    parse_args(argc, argv, &cmd_params);


    /* create relation R */
    fprintf(stdout,
            "[INFO ] Creating relation R with size = %.3lf MiB, #tuples = %d : ",
            (double) sizeof(tuple_t) * cmd_params.r_size/1024.0/1024.0,
            cmd_params.r_size);
    fflush(stdout);

    seed_generator(cmd_params.r_seed);

    /* to pass information to the create_relation methods */
    numalocalize = cmd_params.basic_numa;
    nthreads     = cmd_params.nthreads;

    if(cmd_params.fullrange_keys)
    {
        create_relation_nonunique(&relR, cmd_params.r_size, INT_MAX);
    }
    else if(cmd_params.nonunique_keys)
    {
        create_relation_nonunique(&relR, cmd_params.r_size, cmd_params.r_size);
    }
    else
    {
        create_relation_pk(&relR, cmd_params.r_size);
    }
    printf("OK \n");


    /* create relation S */
    fprintf(stdout,
            "[INFO ] Creating relation S with size = %.3lf MiB, #tuples = %d : ",
            (double) sizeof(tuple_t) * cmd_params.s_size/1024.0/1024.0,
            cmd_params.s_size);
    fflush(stdout);

    seed_generator(cmd_params.s_seed);

    if(cmd_params.fullrange_keys)
    {
        create_relation_fk_from_pk(&relS, &relR, cmd_params.s_size);
    }
    else if(cmd_params.nonunique_keys)
    {
        /* use size of R as the maxid */
        create_relation_nonunique(&relS, cmd_params.s_size, cmd_params.r_size);
    }
    else
    {
        /* if r_size == s_size then equal-dataset, else non-equal dataset */

        if(cmd_params.skew > 0)
        {
            /* S is skewed */
            create_relation_zipf(&relS, cmd_params.s_size,
                                 cmd_params.r_size, cmd_params.skew);
        }
        else
        {
            /* S is uniform foreign key */
            create_relation_fk(&relS, cmd_params.s_size, cmd_params.r_size);
        }
    }
    printf("OK \n");


    /* Run the selected join algorithm */
    printf("[INFO ] Running join algorithm %s ...\n", cmd_params.algo->name);

    results = cmd_params.algo->joinAlgo(&relR, &relS, cmd_params.nthreads);

    printf("[INFO ] Results = %llu. DONE.\n", results);

    /* clean-up */
    delete_relation(&relR);
    delete_relation(&relS);

    return 0;
}

/* command line handling functions */
void
print_help(char * progname)
{
    printf("Usage: %s [options]\n", progname);

    printf("\
    Join algorithm selection, algorithms : RJ, PRO, PRH, PRHO, NPO, NPO_st    \n\
       -a --algo=<name>    Run the hash join algorithm named <name> [PRO]     \n\
                                                                              \n\
    Other join configuration options, with default values in [] :             \n\
       -n --nthreads=<N>  Number of threads to use <N> [2]                    \n\
       -r --r-size=<R>    Number of tuples in build relation R <R> [128000000]\n\
       -s --s-size=<S>    Number of tuples in probe relation S <S> [128000000]\n\
       -x --r-seed=<x>    Seed value for generating relation R <x> [12345]    \n\
       -y --s-seed=<y>    Seed value for generating relation S <y> [54321]    \n\
       -z --skew=<z>      Zipf skew parameter for probe relation S <z> [0.0]  \n\
       --non-unique       Use non-unique (duplicated) keys in input relations \n\
       --full-range       Spread keys in relns. in full 32-bit integer range  \n\
       --basic-numa       Numa-localize relations to threads (Experimental)   \n\
                                                                              \n\
                                                                              \n\
    Basic user options                                                        \n\
        -h --help         Show this message                                   \n\
        --verbose         Be more verbose -- show misc extra info             \n\
        --version         Show version                                        \n\
    \n");
}

void
print_version()
{
    printf("\n%s\n", PACKAGE_STRING);
    printf("Copyright (c) 2012, ETH Zurich, Systems Group.\n");
    printf("http://www.systems.ethz.ch/projects/paralleljoins\n\n");
}

static char *
mystrdup (const char *s)
{
    char *ss = (char*) malloc (strlen (s) + 1);

    if (ss != NULL)
        memcpy (ss, s, strlen(s) + 1);

    return ss;
}

void
parse_args(int argc, char ** argv, param_t * cmd_params)
{

    int c, i, found;
    /* Flag set by ‘--verbose’. */
    static int verbose_flag;
    static int nonunique_flag;
    static int fullrange_flag;
    static int basic_numa;

    while(1)
    {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose",    no_argument,    &verbose_flag,   1},
            {"brief",      no_argument,    &verbose_flag,   0},
            {"non-unique", no_argument,    &nonunique_flag, 1},
            {"full-range", no_argument,    &fullrange_flag, 1},
            {"basic-numa", no_argument,    &basic_numa, 1},
            {"help",       no_argument,    0, 'h'},
            {"version",    no_argument,    0, 'v'},
            /* These options don't set a flag.
               We distinguish them by their indices. */
            {"algo",    required_argument, 0, 'a'},
            {"nthreads",required_argument, 0, 'n'},
            {"perfconf",required_argument, 0, 'p'},
            {"r-size",  required_argument, 0, 'r'},
            {"s-size",  required_argument, 0, 's'},
            {"perfout", required_argument, 0, 'o'},
            {"r-seed",  required_argument, 0, 'x'},
            {"s-seed",  required_argument, 0, 'y'},
            {"skew",    required_argument, 0, 'z'},
            {0, 0, 0, 0}
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long (argc, argv, "a:n:p:r:s:o:x:y:z:hv",
                         long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;
        switch (c)
        {
        case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf ("option %s", long_options[option_index].name);
            if (optarg)
                printf (" with arg %s", optarg);
            printf ("\n");
            break;

        case 'a':
            i = 0;
            found = 0;
            while(algos[i].joinAlgo)
            {
                if(strcmp(optarg, algos[i].name) == 0)
                {
                    cmd_params->algo = &algos[i];
                    found = 1;
                    break;
                }
                i++;
            }

            if(found == 0)
            {
                printf("[ERROR] Join algorithm named `%s' does not exist!\n",
                       optarg);
                print_help(argv[0]);
                exit(EXIT_SUCCESS);
            }
            break;

        case 'h':
        case '?':
            /* getopt_long already printed an error message. */
            print_help(argv[0]);
            exit(EXIT_SUCCESS);
            break;

        case 'v':
            print_version();
            exit(EXIT_SUCCESS);
            break;

        case 'n':
            cmd_params->nthreads = atoi(optarg);
            break;

        case 'p':
            cmd_params->perfconf = mystrdup(optarg);
            break;

        case 'r':
            cmd_params->r_size = atoi(optarg);
            break;

        case 's':
            cmd_params->s_size = atoi(optarg);
            break;

        case 'o':
            cmd_params->perfout = mystrdup(optarg);
            break;

        case 'x':
            cmd_params->r_seed = atoi(optarg);
            break;

        case 'y':
            cmd_params->s_seed = atoi(optarg);
            break;

        case 'z':
            cmd_params->skew = atof(optarg);
            break;

        default:
            break;
        }
    }

    /* if (verbose_flag) */
    /*     printf ("verbose flag is set \n"); */

    cmd_params->nonunique_keys = nonunique_flag;
    cmd_params->verbose        = verbose_flag;
    cmd_params->fullrange_keys = fullrange_flag;
    cmd_params->basic_numa     = basic_numa;

    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf ("non-option arguments: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        printf ("\n");
    }
}
