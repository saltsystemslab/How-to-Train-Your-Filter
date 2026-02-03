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

WrappedStackedFilter *StackedFilterCreate(size_t total_size,
                                   const void *positives,
                                   size_t num_positives,
                                   const void *negatives,
                                   size_t num_negatives,
                                   const double *cdf,
                                   size_t cdf_size,
                                   double insert_capacity);

void StackedFilterDestroy(WrappedStackedFilter *wrapped_filter);

bool StackedFilterLookupElement(WrappedStackedFilter *wrapped_filter, uint64_t *element);

void StackedFilterInsertPositiveElement(WrappedStackedFilter *wrapped_filter, uint64_t *element);
#ifdef __cplusplus
}
#endif

#endif