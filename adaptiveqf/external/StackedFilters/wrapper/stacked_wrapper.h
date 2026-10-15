/**
 * Declares a C API wrapper around the C++ StackedFilters class.
 */

#ifndef STACKED_WRAPPER_H
#define STACKED_WRAPPER_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
    #include "StackedFilter/StackedFilter.h"
extern "C" {
# else
    typedef struct WrappedStackedFilter WrappedStackedFilter;
#endif

// optimize_ns (if not NULL) receives the time spent choosing the layers; insert_capacity must be 0.
WrappedStackedFilter *StackedFilterCreate(size_t total_size,
                                   const void *positives,
                                   size_t num_positives,
                                   const void *negatives,
                                   size_t num_negatives,
                                   const double *cdf,
                                   size_t cdf_size,
                                   double insert_capacity,
                                   uint64_t *optimize_ns);

// Expected FPR of a stacked filter of total_size bits, as predicted by the optimizer that the size
// constructor uses to choose its layers (over the query distribution described by cdf). Does not build a filter.
double StackedFilterExpectedFPR(size_t total_size,
                                size_t num_positives,
                                const double *cdf,
                                size_t cdf_size,
                                double insert_capacity);

// Actual size of the filter's layers, in bits.
size_t StackedFilterGetSize(WrappedStackedFilter *wrapped_filter);

void StackedFilterDestroy(WrappedStackedFilter *wrapped_filter);

bool StackedFilterLookupElement(WrappedStackedFilter *wrapped_filter, uint64_t *element);

void StackedFilterInsertPositiveElement(WrappedStackedFilter *wrapped_filter, uint64_t *element);
#ifdef __cplusplus
}
#endif

#endif