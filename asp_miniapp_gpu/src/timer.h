#include <time.h>

struct timer {
    struct timespec start;
    const clockid_t clock_id = CLOCK_MONOTONIC;//_RAW;
    void tick() { clock_gettime(clock_id, &start); }
    double tock() { struct timespec t; clock_gettime(clock_id, &t); return t.tv_sec - start.tv_sec + (t.tv_nsec - start.tv_nsec)/1e9; }
};