/*
 * Copyright ¬© 2009 inria.  All rights reserved.
 * Copyright ¬© 2009 Universit√© Bordeaux
 * See COPYING in top-level directory.
 */

#ifndef HWLOC_PORT_DARWIN_SYS_SYSCTL_H
#define HWLOC_PORT_DARWIN_SYS_SYSCTL_H

extern int sysctl(int *name, int name_len, void *oldp, size_t *oldlenp, void *newp, size_t newlen);
extern int sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);

#define CTL_HW 6
#define HW_PHYSMEM 5

#endif /* HWLOC_PORT_DARWIN_SYS_SYSCTL_H */
