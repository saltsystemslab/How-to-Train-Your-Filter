#include "StackedFilter.h"
#include <cstdlib>

extern "C" {

struct WrappedStackedFilter {
    StackedFilter<BloomFilterLayer, IntElement> *filter;
};

WrappedStackedFilter *StackedFilterCreate(size_t total_size,
                                   const void *positives,
                                   size_t num_positives,
                                   const void *negatives,
                                   size_t num_negatives,
                                   const double *cdf,
                                   size_t cdf_size,
                                   double insert_capacity) {
    std::vector<IntElement> positive_vec;
    std::vector<IntElement> negative_vec;
    std::vector<double> cdf_vec;

    for (size_t i = 0; i < num_positives; i++) {
        positive_vec.emplace_back(static_cast<const uint64_t *>(positives)[i]);
    }
    for (size_t i = 0; i < num_negatives; i++) {
        negative_vec.emplace_back(static_cast<const uint64_t *>(negatives)[i]);
    }
    for (size_t i = 0; i < cdf_size; i++) {
        cdf_vec.push_back(cdf[i]);
    }
    WrappedStackedFilter *wrapped_filter = new WrappedStackedFilter;
    wrapped_filter->filter = new StackedFilter<BloomFilterLayer, IntElement>(
            total_size, positive_vec, negative_vec, cdf_vec, insert_capacity);
    return wrapped_filter;
}

void StackedFilterDestroy(WrappedStackedFilter *wrapped_filter) {
    delete wrapped_filter->filter;
    delete wrapped_filter;
}

bool StackedFilterLookupElement(WrappedStackedFilter *wrapped_filter, uint64_t *element) {
    return wrapped_filter->filter->LookupElement(*element);
}

void StackedFilterInsertPositiveElement(WrappedStackedFilter *wrapped_filter, uint64_t *element) {
    wrapped_filter->filter->InsertPositiveElement(*element);
}

}