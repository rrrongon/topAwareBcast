/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

/* Copyright (c) 2001-2019, The Ohio State University. All rights
 * reserved.
 *
 * This file is part of the MVAPICH2 software package developed by the
 * team members of The Ohio State University's Network-Based Computing
 * Laboratory (NBCL), headed by Professor Dhabaleswar K. (DK) Panda.
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level MVAPICH2 directory.
 *
 */
#include <strings.h>

#include "mpiimpl.h"
#include "mpi_init.h"

#if defined(CHANNEL_MRAIL) || defined(CHANNEL_PSM)
#include "coll_shmem.h"
#endif


/*@
 * Added by rubayet
 * @*/

#include <stdio.h>
#include <stdlib.h>
#define __USE_GNU
#include <sched.h>
#include <errno.h>
#include <unistd.h>

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <inttypes.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <fcntl.h>
#include <numaif.h>
#include <numa.h>

/*@
 *Added by rubayet
 * 
 * @*/

/*
=== BEGIN_MPI_T_CVAR_INFO_BLOCK ===

categories:
    - name        : THREADS
      description : multi-threading cvars

cvars:
    - name        : MPIR_CVAR_ASYNC_PROGRESS
      category    : THREADS
      type        : boolean
      default     : false
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        If set to true, MPICH will initiate an additional thread to
        make asynchronous progress on all communication operations
        including point-to-point, collective, one-sided operations and
        I/O.  Setting this variable will automatically increase the
        thread-safety level to MPI_THREAD_MULTIPLE.  While this
        improves the progress semantics, it might cause a small amount
        of performance overhead for regular MPI operations.  The user
        is encouraged to leave one or more hardware threads vacant in
        order to prevent contention between the application threads
        and the progress thread(s).  The impact of oversubscription is
        highly system dependent but may be substantial in some cases,
        hence this recommendation.

    - name        : MPIR_CVAR_DEFAULT_THREAD_LEVEL
      category    : THREADS
      type        : string
      default     : "MPI_THREAD_SINGLE"
      class       : device
      verbosity   : MPI_T_VERBOSITY_USER_BASIC
      scope       : MPI_T_SCOPE_ALL_EQ
      description : >-
        Sets the default thread level to use when using MPI_INIT. This variable
        is case-insensitive.

=== END_MPI_T_CVAR_INFO_BLOCK ===
*/

/* -- Begin Profiling Symbol Block for routine MPI_Init */
#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Init = PMPI_Init
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Init  MPI_Init
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Init as PMPI_Init
#elif defined(HAVE_WEAK_ATTRIBUTE)
int MPI_Init(int *argc, char ***argv) __attribute__((weak,alias("PMPI_Init")));
#endif
/* -- End Profiling Symbol Block */

/* Define MPICH_MPI_FROM_PMPI if weak symbols are not supported to build
   the MPI routines */
#ifndef MPICH_MPI_FROM_PMPI
#undef MPI_Init
#define MPI_Init PMPI_Init

/* Fortran logical values. extern'd in mpiimpl.h */
/* MPI_Fint MPIR_F_TRUE, MPIR_F_FALSE; */

/* Any internal routines can go here.  Make them static if possible */

/* must go inside this #ifdef block to prevent duplicate storage on darwin */
int MPIR_async_thread_initialized = 0;
#endif

#undef FUNCNAME
#define FUNCNAME MPI_Init

/*@
   MPI_Init - Initialize the MPI execution environment

Input Parameters:
+  argc - Pointer to the number of arguments 
-  argv - Pointer to the argument vector

Thread and Signal Safety:
This routine must be called by one thread only.  That thread is called
the `main thread` and must be the thread that calls 'MPI_Finalize'.

Notes:
   The MPI standard does not say what a program can do before an 'MPI_INIT' or
   after an 'MPI_FINALIZE'.  In the MPICH implementation, you should do
   as little as possible.  In particular, avoid anything that changes the
   external state of the program, such as opening files, reading standard
   input or writing to standard output.

Notes for C:
    As of MPI-2, 'MPI_Init' will accept NULL as input parameters. Doing so
    will impact the values stored in 'MPI_INFO_ENV'.

Notes for Fortran:
The Fortran binding for 'MPI_Init' has only the error return
.vb
    subroutine MPI_INIT( ierr )
    integer ierr
.ve

.N Errors
.N MPI_SUCCESS
.N MPI_ERR_INIT

.seealso: MPI_Init_thread, MPI_Finalize
@*/



/*@
 *
 *Added by rubayet
 * @*/

typedef struct numa_node_info
{   
    size_t core_cnt;
    int numa_node;
    int *cores;
    
} numa_t;


long long int print_duration(struct timespec *b, struct timespec *c)
{
        long long r = c->tv_nsec - b->tv_nsec;
        r += ((long long)(c->tv_sec - b->tv_sec) ) * 1000000000;
        return r;
}

void node_distance(int numa_cnt, int *numa_dist_matrix){
        int src_numa, target_numa, pos, numa_count, numa_dist;
        numa_count = numa_cnt;
        for(src_numa=0; src_numa < numa_count; src_numa++){
                for(target_numa=0; target_numa< numa_count; target_numa++){
                        pos = (src_numa * numa_count) + target_numa;
                        numa_dist = numa_distance(src_numa, target_numa);
                        printf("src: %d, dest: %d: %d\n", src_numa, target_numa, numa_dist);
                        *(numa_dist_matrix + pos)= numa_dist;
                }
        }
}

void print_numa_distance(int numa_cnt, int *numa_dist_matrix){
        int src_numa, target_numa, pos, numa_count, numa_dist;
        numa_count = numa_cnt;
        for(src_numa=0; src_numa < numa_count; src_numa++){
                printf("From numa %d: ", src_numa);
                for(target_numa=0; target_numa<numa_count; target_numa++){
                        pos = (src_numa * numa_count) + target_numa;
                        numa_dist = *(numa_dist_matrix + pos);
                        printf(" %d: %d",target_numa, numa_dist);
                }
                printf("\n");
        }
}

void print_numa_bitmask(struct bitmask *bm)
{
        size_t i;
    for(i=0;i<bm->size;++i)
    {
        printf("%d ", numa_bitmask_isbitset(bm, i));
    }
}

int count_cpu_onnode(struct bitmask *bm)
{
        size_t i;
        int cnt=0;
    for(i=0;i<bm->size;++i)
    {
         if(numa_bitmask_isbitset(bm, i))
                 cnt++;
    }
    return cnt;
}

void get_cores_onnode(int *cores_onnode, int cnt, struct bitmask *bm){
        size_t i;
        cnt=0;
    for(i=0;i<bm->size;++i){
         if(numa_bitmask_isbitset(bm, i)){
                 *(cores_onnode+cnt) = i;
                 cnt++;
        }
    }
}

void show_numa_infos(numa_t *numa_infos, int numa_cnt){
        int i,j;
        for(i=0;i< numa_cnt; i++){
                printf("numa node: %d, core_cnt: %d\n", numa_infos[i].numa_node, numa_infos[i].core_cnt);
                int * cores = numa_infos[i].cores;
                for(j=0; j<numa_infos[i].core_cnt; j++){
                        printf("%d, ", cores[j]);
                }
                printf("\n");
        }
}

int numberOfProcessors, numcpus, numa_cnt, *numa_dist_matrix, cpu_cnt_onnode;
numa_t * numa_infos;
struct bitmask* bm;

/*@
 *Added by Rubayet
 * @*/


int MPI_Init( int *argc, char ***argv )
{
    static const char FCNAME[] = "MPI_Init";
    int mpi_errno = MPI_SUCCESS;
    int rc ATTRIBUTE((unused));
    int threadLevel, provided;
    MPID_MPI_INIT_STATE_DECL(MPID_STATE_MPI_INIT);

    /* Handle mpich_state in case of Re-init */
    if (OPA_load_int(&MPIR_Process.mpich_state) == MPICH_POST_FINALIZED) {
        OPA_store_int(&MPIR_Process.mpich_state, MPICH_PRE_INIT);
    }
    rc = MPID_Wtime_init();
#ifdef USE_DBG_LOGGING
    MPIU_DBG_PreInit( argc, argv, rc );
#endif

    MPID_MPI_INIT_FUNC_ENTER(MPID_STATE_MPI_INIT);
#if defined(CHANNEL_PSM)
    MV2_Read_env_vars();
#endif /* defined(CHANNEL_PSM) */

#   ifdef HAVE_ERROR_CHECKING
    {
        MPID_BEGIN_ERROR_CHECKS;
        {
            if (OPA_load_int(&MPIR_Process.mpich_state) != MPICH_PRE_INIT) {
                mpi_errno = MPIR_Err_create_code( MPI_SUCCESS, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER,
						  "**inittwice", NULL );
	    }
            if (mpi_errno) goto fn_fail;
        }
        MPID_END_ERROR_CHECKS;
    }
#   endif /* HAVE_ERROR_CHECKING */

    /* ... body of routine ... */

    /* Temporarily disable thread-safety.  This is needed because the
     * mutexes are not initialized yet, and we don't want to
     * accidentally use them before they are initialized.  We will
     * reset this value once it is properly initialized. */
#if defined MPICH_IS_THREADED
    MPIR_ThreadInfo.isThreaded = 0;
#endif /* MPICH_IS_THREADED */

    MPIR_T_env_init();

    if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_MULTIPLE"))
        threadLevel = MPI_THREAD_MULTIPLE;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_SERIALIZED"))
        threadLevel = MPI_THREAD_SERIALIZED;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_FUNNELED"))
        threadLevel = MPI_THREAD_FUNNELED;
    else if (!strcasecmp(MPIR_CVAR_DEFAULT_THREAD_LEVEL, "MPI_THREAD_SINGLE"))
        threadLevel = MPI_THREAD_SINGLE;
    else {
        MPL_error_printf("Unrecognized thread level %s\n", MPIR_CVAR_DEFAULT_THREAD_LEVEL);
        exit(1);
    }

    /* If the user requested for asynchronous progress, request for
     * THREAD_MULTIPLE. */
    if (MPIR_CVAR_ASYNC_PROGRESS)
        threadLevel = MPI_THREAD_MULTIPLE;

    mpi_errno = MPIR_Init_thread( argc, argv, threadLevel, &provided );
    if (mpi_errno != MPI_SUCCESS) goto fn_fail;

    if (MPIR_CVAR_ASYNC_PROGRESS) {
        if (provided == MPI_THREAD_MULTIPLE) {
            mpi_errno = MPIR_Init_async_thread();
            if (mpi_errno) goto fn_fail;

            MPIR_async_thread_initialized = 1;
        }
        else {
            printf("WARNING: No MPI_THREAD_MULTIPLE support (needed for async progress)\n");
        }
    }

	/*@
	 * Added by rubayet
	 * @*/
        int i,j,k;
	numberOfProcessors =sysconf(_SC_NPROCESSORS_ONLN);
	numcpus = numa_num_configured_cpus();
        numa_cnt = numa_max_node() + 1;

	numa_dist_matrix = (int*)malloc(numa_cnt * numa_cnt);
        node_distance(numa_cnt, numa_dist_matrix);

	bm = numa_bitmask_alloc(numcpus);
        printf("Max number of NUMA nodes %d\n", numa_max_node());

	numa_infos= (numa_t *) malloc(numa_cnt* sizeof(numa_t));
        for (i=0;i<=numa_max_node();++i){
                numa_node_to_cpus(i, bm);
                printf("Numa nodes %d CPU status:",i);
                print_numa_bitmask(bm);
                cpu_cnt_onnode = count_cpu_onnode(bm);
                int * cores_onnode = (int *)malloc(cpu_cnt_onnode* sizeof(int));
                get_cores_onnode(cores_onnode,cpu_cnt_onnode,bm);
                numa_t temp_numa;
                temp_numa.core_cnt = cpu_cnt_onnode;
                temp_numa.numa_node = i;
                temp_numa.cores = cores_onnode;
                numa_infos[i] = temp_numa;
                printf("on node %d, cpu count: %d\n", i, cpu_cnt_onnode);
        }
        numa_bitmask_free(bm);

	show_numa_infos(numa_infos, numa_cnt);
    	/*@
	 * Added by rubayet
	 * @*/

    /* initialize the two level communicator for MPI_COMM_WORLD  */
    if (mv2_use_osu_collectives && 
            mv2_enable_shmem_collectives) {

       MPID_Comm *comm_ptr = NULL;
       MPID_Comm_get_ptr(MPI_COMM_WORLD, comm_ptr);
       int flag=0; 
       PMPI_Comm_test_inter(comm_ptr->handle, &flag);

       if(flag == 0 && comm_ptr->dev.ch.shmem_coll_ok == 0 &&
               comm_ptr->local_size < mv2_two_level_comm_early_init_threshold &&
               check_split_comm(pthread_self())) { 

            disable_split_comm(pthread_self());
            mpi_errno = create_2level_comm(comm_ptr->handle, comm_ptr->local_size, comm_ptr->rank);
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
            enable_split_comm(pthread_self());
            if(mpi_errno) {
               MPIR_ERR_POP(mpi_errno);
            }
       } 
    }

    /* ... end of body of routine ... */
    MPID_MPI_INIT_FUNC_EXIT(MPID_STATE_MPI_INIT);
    return mpi_errno;

  fn_fail:
    /* --BEGIN ERROR HANDLING-- */
#   ifdef HAVE_ERROR_REPORTING
    {
	mpi_errno = MPIR_Err_create_code(
	    mpi_errno, MPIR_ERR_RECOVERABLE, FCNAME, __LINE__, MPI_ERR_OTHER, 
	    "**mpi_init", "**mpi_init %p %p", argc, argv);
    }
#   endif
    mpi_errno = MPIR_Err_return_comm( 0, FCNAME, mpi_errno );
    return mpi_errno;
    /* --END ERROR HANDLING-- */
}
