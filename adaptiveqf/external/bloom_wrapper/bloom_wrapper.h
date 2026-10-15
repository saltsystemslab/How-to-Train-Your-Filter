/**
 * Declares a C API wrapper around the C++ Bloom filter used by CascadedLBF
 * (bloomfilter::BloomFilter from CascadedLBF/thirdparty/fastfilter_cpp/src/bloom/bloom.h).
 */

#ifndef BLOOM_WRAPPER_H
#define BLOOM_WRAPPER_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WrappedBloomFilter WrappedBloomFilter;

// Creates a Bloom filter sized for num_keys keys at bits_per_key bits each, with its hash seeded by seed.
WrappedBloomFilter *BloomFilterCreate(size_t num_keys, double bits_per_key, uint64_t seed);

void BloomFilterDestroy(WrappedBloomFilter *wrapped_filter);

void BloomFilterInsertElement(WrappedBloomFilter *wrapped_filter, uint64_t element);

bool BloomFilterLookupElement(WrappedBloomFilter *wrapped_filter, uint64_t element);

// Size of the filter's bit array in bytes.
size_t BloomFilterSizeInBytes(WrappedBloomFilter *wrapped_filter);

#ifdef __cplusplus
}
#endif

#endif
