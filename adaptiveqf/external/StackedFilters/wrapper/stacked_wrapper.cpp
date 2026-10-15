#include "StackedFilter.h"
#include <cstdlib>
#include <ctime>

extern "C" {

struct WrappedStackedFilter {
    StackedFilter<BloomFilterLayer, BigIntElement> *filter;
};

WrappedStackedFilter *StackedFilterCreate(size_t total_size,
                                   const void *positives,
                                   size_t num_positives,
                                   const void *negatives,
                                   size_t num_negatives,
                                   const double *cdf,
                                   size_t cdf_size,
                                   double insert_capacity,
                                   uint64_t *optimize_ns) {
    std::vector<BigIntElement> positive_vec;
    std::vector<BigIntElement> negative_vec;
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
    // The StackedFilter size constructor, split into choosing the layers (its optimizer) and building the filter
    // from them, so the optimization can be timed separately. Only an insert capacity of 0 is supported, which is
    // what the constructor that takes the layers uses.
    static constexpr double kEpsilonError = .000001;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t start = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
    uint64_t num_chosen_negatives;
    std::vector<double> layer_fprs;
    std::tie(num_chosen_negatives, layer_fprs) = optimizeStackedFilterBloom(
            static_cast<double>(total_size) / positive_vec.size(), positive_vec.size() * (1 + insert_capacity),
            kEpsilonError, cdf_vec);
    std::vector<uint32> integral_parameters;
    for (auto layer_fpr : layer_fprs) {
        integral_parameters.push_back(round(-log2(layer_fpr)));
    }
    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (optimize_ns) *optimize_ns = (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec - start;
    num_chosen_negatives = std::min<uint64_t>(negative_vec.size(), num_chosen_negatives);
    std::vector<BigIntElement> chosen_negatives(negative_vec.begin(), negative_vec.begin() + num_chosen_negatives);
    WrappedStackedFilter *wrapped_filter = new WrappedStackedFilter;
    wrapped_filter->filter = new StackedFilter<BloomFilterLayer, BigIntElement>(
            layer_fprs, integral_parameters, positive_vec, chosen_negatives);
    return wrapped_filter;
}

// FPR of one Bloom layer of the given size holding num_elements, as built by InitStackedAMQ.
// With 0 hashes the layer's lookup loop never runs, so it reports every element as present.
static double built_layer_fpr(size_t size_bits, double num_elements, int num_hashes) {
    if (num_hashes <= 0) return 1.0;
    return std::pow(1.0 - std::exp(-num_hashes * num_elements / size_bits), num_hashes);
}

double StackedFilterExpectedFPR(size_t total_size,
                                size_t num_positives,
                                const double *cdf,
                                size_t cdf_size,
                                double insert_capacity) {
    // Same optimizer call (and constants) as the StackedFilter size constructor, without building the filter.
    static constexpr double kEpsilonError = .000001;
    std::vector<double> cdf_vec(cdf, cdf + cdf_size);
    uint64_t num_chosen_negatives;
    std::vector<double> layer_fprs;
    std::tie(num_chosen_negatives, layer_fprs) = optimizeStackedFilterBloom(
            static_cast<double>(total_size) / num_positives, num_positives * (1 + insert_capacity),
            kEpsilonError, cdf_vec);
    num_chosen_negatives = std::min<uint64_t>(num_chosen_negatives, cdf_size);

    // Model the filter as InitStackedAMQ builds it rather than the optimizer's continuous estimate:
    // integer hash counts (round(-log2 fpr), which can be 0), layer sizes from SizeFunction with its
    // 2000-bit floor, and the expected number of elements reaching each layer.
    int num_layers = layer_fprs.size();
    double num_pos_fp = num_positives;          // positives inserted into the next even layer
    double num_neg_fp = num_chosen_negatives;   // known negatives inserted into the next odd layer
    double known_fpr = 1.0;     // a known negative passes every even layer it is tested against
    double unknown_fpr = 0.0;   // an unknown negative is accepted when an odd layer misses it
    double reach = 1.0;         // probability an unknown negative reaches the current layer
    for (int i = 0; i < num_layers; i++) {
        bool positive_layer = (i % 2) == 0;
        double count = positive_layer ? num_pos_fp * (1 + insert_capacity) : num_neg_fp;
        size_t size = BloomFilterLayer<BigIntElement>::SizeFunction(layer_fprs[i], static_cast<size_t>(count + 0.5));
        if (size < 2000) size = 2000;
        int num_hashes = static_cast<int>(std::round(-std::log2(layer_fprs[i])));
        double inserted = positive_layer ? num_pos_fp : num_neg_fp;
        double fpr = built_layer_fpr(size, inserted, num_hashes);
        if (positive_layer) {
            known_fpr *= fpr;
            num_neg_fp *= fpr;      // known negatives that pass this layer go into the next one
        } else {
            unknown_fpr += reach * (1.0 - fpr);
            num_pos_fp *= fpr;      // positives that pass this negative layer go into the next one
        }
        reach *= fpr;
    }
    unknown_fpr += reach;   // passed every layer
    if (num_layers == 1 || num_chosen_negatives == 0) {
        // no negative layers: every negative is filtered by the first layer alone
        return unknown_fpr;
    }
    double psi = cdf_vec[num_chosen_negatives - 1];
    return psi * known_fpr + (1 - psi) * unknown_fpr;
}

size_t StackedFilterGetSize(WrappedStackedFilter *wrapped_filter) {
    return wrapped_filter->filter->GetSize();
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