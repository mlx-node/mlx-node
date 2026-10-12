// macOS development probe; does not run in the inference process.
// Build: xcrun clang -O2 scripts/tts/process-memory.c -o /tmp/process-memory
// Use: /tmp/process-memory <pid>
// Footprint and RSS overlap; never add them. Peak footprint is the kernel's
// cumulative high-water mark as of this sample; a later exit-time peak can be missed.
#include <libproc.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  char *end = NULL;
  long pid = strtol(argv[1], &end, 10);
  if (*end || pid < 1 || pid > INT_MAX) return 2;
  struct rusage_info_v4 info = {0};
  if (proc_pid_rusage((int)pid, RUSAGE_INFO_V4, (rusage_info_t *)&info)) return 1;
  printf("{\"physicalFootprintBytes\":%llu,\"peakPhysicalFootprintBytes\":%llu,"
         "\"residentBytes\":%llu,\"wiredBytes\":%llu}\n",
         (unsigned long long)info.ri_phys_footprint,
         (unsigned long long)info.ri_lifetime_max_phys_footprint,
         (unsigned long long)info.ri_resident_size,
         (unsigned long long)info.ri_wired_size);
  return 0;
}
