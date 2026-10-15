#include "bloom_wrapper.h"
#include "bloom/bloom.h"

extern "C" {

struct WrappedBloomFilter {
    bloomfilter::BloomFilter<uint64_t, false> *filter;
};

WrappedBloomFilter *BloomFilterCreate(size_t num_keys, double bits_per_key, uint64_t seed) {
    WrappedBloomFilter *wrapped_filter = new WrappedBloomFilter;
    wrapped_filter->filter = new bloomfilter::BloomFilter<uint64_t, false>(num_keys, bits_per_key);
    // the hasher seeds itself from std::random_device; use the given seed so runs are reproducible
    wrapped_filter->filter->hasher.setSeed(seed);
    return wrapped_filter;
}

void BloomFilterDestroy(WrappedBloomFilter *wrapped_filter) {
    // BloomFilter's destructor is commented out upstream, so its bit array is freed here
    delete[] wrapped_filter->filter->data;
    delete wrapped_filter->filter;
    delete wrapped_filter;
}

void BloomFilterInsertElement(WrappedBloomFilter *wrapped_filter, uint64_t element) {
    wrapped_filter->filter->Add(element);
}

bool BloomFilterLookupElement(WrappedBloomFilter *wrapped_filter, uint64_t element) {
    return wrapped_filter->filter->Contain(element) == bloomfilter::Ok;
}

size_t BloomFilterSizeInBytes(WrappedBloomFilter *wrapped_filter) {
    return wrapped_filter->filter->SizeInBytes();
}

}
