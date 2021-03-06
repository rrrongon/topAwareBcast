/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#ifndef MPIU_TIMER_H_INCLUDED
#define MPIU_TIMER_H_INCLUDED

#include "mpichconf.h"
#include <stdlib.h>

#if defined (HAVE_UNISTD_H)
#include <unistd.h>
#if defined (NEEDS_USLEEP_DECL)
int usleep(useconds_t usec);
#endif
#endif

#include "mpl.h"

/*
 * This include file provide the definitions that are necessary to use the
 * timer calls, including the definition of the time stamp type and
 * any inlined timer calls.
 *
 * The include file timerconf.h (created by autoheader from configure.ac)
 * is needed only to build the function versions of the timers.
 */
/* Include the appropriate files */
#define MPIU_GETHRTIME               1
#define MPIU_CLOCK_GETTIME           2
#define MPIU_GETTIMEOFDAY            3
#define MPIU_LINUX86_CYCLE           4
#define MPIU_LINUXALPHA_CYCLE        5
#define MPIU_QUERYPERFORMANCECOUNTER 6
#define MPIU_WIN86_CYCLE             7
#define MPIU_GCC_IA64_CYCLE          8
/* The value "MPIU_DEVICE" means that the ADI device provides the timer */
#define MPIU_DEVICE                  9
#define MPIU_WIN64_CYCLE             10
#define MPIU_MACH_ABSOLUTE_TIME      11
#define MPICH_TIMER_KIND MPIU_CLOCK_GETTIME

#if MPICH_TIMER_KIND == MPIU_GETHRTIME
#include <sys/time.h>
#elif MPICH_TIMER_KIND == MPIU_CLOCK_GETTIME
#include <time.h>
#ifdef NEEDS_SYS_TIME_H
/* Some OS'es mistakenly require sys/time.h to get the definition of
   CLOCK_REALTIME (POSIX requires the definition to be in time.h) */
#include <sys/time.h>
#endif
#elif MPICH_TIMER_KIND == MPIU_GETTIMEOFDAY
#include <sys/types.h>
#include <sys/time.h>
#elif MPICH_TIMER_KIND == MPIU_LINUX86_CYCLE
#elif MPICH_TIMER_KIND == MPIU_GCC_IA64_CYCLE
#elif MPICH_TIMER_KIND == MPIU_LINUXALPHA_CYCLE
#elif MPICH_TIMER_KIND == MPIU_QUERYPERFORMANCECOUNTER
#include <winsock2.h>
#include <windows.h>
#elif MPICH_TIMER_KIND == MPIU_MACH_ABSOLUTE_TIME
#include <mach/mach_time.h>
#elif MPICH_TIMER_KIND == MPIU_WIN86_CYCLE
#include <winsock2.h>
#include <windows.h>
#endif

/* Define a time stamp */
typedef struct timespec MPIU_Time_t;

/*
 * Prototypes.  These are defined here so that inlined timer calls can
 * use them, as well as any profiling and timing code that is built into
 * MPICH
 */
/*@
  MPIU_Wtime - Return a time stamp

  Output Parameter:
. timeval - A pointer to an 'MPIU_Wtime_t' variable.

  Notes:
  This routine returns an `opaque` time value.  This difference between two
  time values returned by 'MPIU_Wtime' can be converted into an elapsed time
  in seconds with the routine 'MPIU_Wtime_diff'.

  This routine is defined this way to simplify its implementation as a macro.
  For example, the for Intel x86 and gcc,
.vb
#define MPIU_Wtime(timeval) \
     __asm__ __volatile__("rdtsc" : "=A" (*timeval))
.ve

  For some purposes, it is important
  that the timer calls change the timing of the code as little as possible.
  This form of a timer routine provides for a very fast timer that has
  minimal impact on the rest of the code.

  From a semantic standpoint, this format emphasizes that any particular
  timer value has no meaning; only the difference between two values is
  meaningful.

  Module:
  Timer

  Question:
  MPI-2 allows 'MPI_Wtime' to be a macro.  We should make that easy; this
  version does not accomplish that.
  @*/
void MPIU_Wtime( MPIU_Time_t * timeval);

/*@
  MPIU_Wtime_diff - Compute the difference between two time stamps

  Input Parameters:
. t1, t2 - Two time values set by 'MPIU_Wtime' on this process.


  Output Parameter:
. diff - The different in time between t2 and t1, measured in seconds.

  Note:
  If 't1' is null, then 't2' is assumed to be differences accumulated with
  'MPIU_Wtime_acc', and the output value gives the number of seconds that
  were accumulated.

  Question:
  Instead of handling a null value of 't1', should we have a separate
  routine 'MPIU_Wtime_todouble' that converts a single timestamp to a
  double value?

  Module:
  Timer
  @*/
void MPIU_Wtime_diff( MPIU_Time_t *t1, MPIU_Time_t *t2, double *diff );

/*@
  MPIU_Wtime_acc - Accumulate time values

  Input Parameters:
. t1,t2,t3 - Three time values.  't3' is updated with the difference between
             't2' and 't1': '*t3 += (t2 - t1)'.

  Notes:
  This routine is used to accumulate the time spent with a block of code
  without first converting the time stamps into a particular arithmetic
  type such as a 'double'.  For example, if the 'MPIU_Wtime' routine accesses
  a cycle counter, this routine (or macro) can perform the accumulation using
  integer arithmetic.

  To convert a time value accumulated with this routine, use 'MPIU_Wtime_diff'
  with a 't1' of zero.

  Module:
  Timer
  @*/
void MPIU_Wtime_acc( MPIU_Time_t *t1, MPIU_Time_t *t2, MPIU_Time_t *t3 );

/*@
  MPIU_Wtime_todouble - Converts an MPID timestamp to a double

  Input Parameter:
. timeval - 'MPIU_Time_t' time stamp

  Output Parameter:
. seconds - Time in seconds from an arbitrary (but fixed) time in the past

  Notes:
  This routine may be used to change a timestamp into a number of seconds,
  suitable for 'MPI_Wtime'.

  @*/
void MPIU_Wtime_todouble( MPIU_Time_t *timeval, double *seconds );

/*@
  MPIU_Wtick - Provide the resolution of the 'MPIU_Wtime' timer

  Return value:
  Resolution of the timer in seconds.  In many cases, this is the
  time between ticks of the clock that 'MPIU_Wtime' returns.  In other
  words, the minimum significant difference that can be computed by
  'MPIU_Wtime_diff'.

  Note that in some cases, the resolution may be estimated.  No application
  should expect either the same estimate in different runs or the same
  value on different processes.

  Module:
  Timer
  @*/
double MPIU_Wtick( void );

/*@
  MPIU_Wtime_init - Initialize the timer

  Note:
  This routine should perform any steps needed to initialize the timer.
  In addition, it should set the value of the attribute 'MPI_WTIME_IS_GLOBAL'
  if the timer is known to be the same for all processes in 'MPI_COMM_WORLD'
  (the value is zero by default).

  If any operations need to be performed when the MPI program calls
  'MPI_Finalize' this routine should register a handler with 'MPI_Finalize'
  (see the MPICH Design Document).

  Return Values:
  0 on success.  -1 on Failure.  1 means that the timer may not be used
  until after MPIU_Init completes.  This allows the device to set up the
  timer (first needed for Blue Gene support).

  Module:
  Timer

  @*/
int MPIU_Wtime_init(void);

/* Inlined timers.  Note that any definition of one of the functions
   prototyped above in terms of a macro will simply cause the compiler
   to use the macro instead of the function definition.

   Currently, all except the Windows performance counter timers
   define MPIU_Wtime_init as null; by default, the value of
   MPI_WTIME_IS_GLOBAL is false.
 */

/* MPIUM_Wtime_todouble() is a hack to get a macro version
   of the todouble function.

   The logging library should save the native MPIU_Timer_t
   structure to disk and use the todouble function in the
   post-processsing step to convert the values to doubles.
   */

/* The timer kind is set using AC_SUBST in the MPICH configure */
#define MPICH_TIMER_KIND MPIU_CLOCK_GETTIME

#if MPICH_TIMER_KIND == MPIU_GETHRTIME
#define MPIUM_Wtime_todouble MPIU_Wtime_todouble

#elif MPICH_TIMER_KIND == MPIU_CLOCK_GETTIME
#define MPIUM_Wtime_todouble MPIU_Wtime_todouble

#elif MPICH_TIMER_KIND == MPIU_GETTIMEOFDAY
#define MPIUM_Wtime_todouble MPIU_Wtime_todouble

#elif MPICH_TIMER_KIND == MPIU_LINUX86_CYCLE
/* The rdtsc instruction is not a "serializing" instruction, so the
   processor is free to reorder it.  In order to get more accurate
   timing numbers with rdtsc, we need to put a serializing
   instruction, like cpuid, before rdtsc.  X86_64 architectures have
   the rdtscp instruction which is synchronizing, we use this when we
   can. */
#ifdef LINUX86_CYCLE_RDTSCP
#define MPIU_Wtime(var_ptr) \
    __asm__ __volatile__("push %%rbx ; cpuid ; rdtsc ; pop %%rbx ; shl $32, %%rdx; or %%rdx, %%rax" : "=a" (*var_ptr) : : "ecx", "rdx")
#elif defined(LINUX86_CYCLE_CPUID_RDTSC64)
/* Here we have to save the rbx register for when the compiler is
   generating position independent code (e.g., when it's generating
   shared libraries) */
#define MPIU_Wtime(var_ptr)                                                                     \
     __asm__ __volatile__("push %%rbx ; cpuid ; rdtsc ; pop %%rbx" : "=A" (*var_ptr) : : "ecx")
#elif defined(LINUX86_CYCLE_CPUID_RDTSC32)
/* Here we have to save the ebx register for when the compiler is
   generating position independent code (e.g., when it's generating
   shared libraries) */
#define MPIU_Wtime(var_ptr)                                                                     \
     __asm__ __volatile__("push %%ebx ; cpuid ; rdtsc ; pop %%ebx" : "=A" (*var_ptr) : : "ecx")
#elif defined(LINUX86_CYCLE_RDTSC)
/* The configure test using cpuid must have failed, try just rdtsc by itself */
#define MPIU_Wtime(var_ptr) __asm__ __volatile__("rdtsc" : "=A" (*var_ptr))
#else
#error Dont know which Linux timer to use
#endif

extern double MPIU_Seconds_per_tick;
#define MPIUM_Wtime_todouble(t, d) *d = (double)*t * MPIU_Seconds_per_tick

#elif MPICH_TIMER_KIND == MPIU_GCC_IA64_CYCLE
#ifdef __INTEL_COMPILER
#include "ia64regs.h"
#define MPIU_Wtime(var_ptr) { MPIU_Time_t t_val;\
	t_val=__getReg(_IA64_REG_AR_ITC); *var_ptr=t_val;}
#else
#define MPIU_Wtime(var_ptr) { MPIU_Time_t t_val;\
	__asm__ __volatile__("mov %0=ar.itc" : "=r" (t_val)); *var_ptr=t_val;}
#endif
extern double MPIU_Seconds_per_tick;
#define MPIUM_Wtime_todouble(t, d) *d = (double)*t * MPIU_Seconds_per_tick

#elif MPICH_TIMER_KIND == MPIU_LINUXALPHA_CYCLE
#define MPIUM_Wtime_todouble MPIU_Wtime_todouble

#elif MPICH_TIMER_KIND == MPIU_QUERYPERFORMANCECOUNTER
#define MPIU_Wtime(var) QueryPerformanceCounter(var)
extern double MPIU_Seconds_per_tick;
#define MPIUM_Wtime_todouble( t, d ) \
  *d = (double)t->QuadPart * MPIU_Seconds_per_tick /* convert to seconds */

#elif MPICH_TIMER_KIND == MPIU_WIN86_CYCLE
#define MPIU_Wtime(var_ptr) \
{ \
    register int *f1 = (int*)var_ptr; \
    __asm cpuid \
    __asm rdtsc \
    __asm mov ecx, f1 \
    __asm mov [ecx], eax \
    __asm mov [ecx + TYPE int], edx \
}
extern double MPIU_Seconds_per_tick;
#define MPIUM_Wtime_todouble(t, d) *d = (double)(__int64)*t * MPIU_Seconds_per_tick
#define MPIUM_Wtime_diff(t1,t2,diff) *diff = (double)((__int64)( *t2 - *t1 )) * MPIU_Seconds_per_tick

#elif MPICH_TIMER_KIND == MPIU_MACH_ABSOLUTE_TIME
#define MPIUM_Wtime_todouble MPIU_Wtime_todouble

#endif

#endif
