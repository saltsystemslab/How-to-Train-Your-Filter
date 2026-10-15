#include <stdio.h>
#include "include/rand_util.h"

// Random 64-bit values (splitmix64), seeded so the queries can be regenerated exactly. rand() alone gives only 31
// random bits, too few to reach the least likely ranks of the larger datasets.
static uint64_t rng_state = 0x5eed2f1b9a3c7d01ULL;
static uint64_t next_random(void) {
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// The cumulative probabilities of the ranks 1..max of a Zipfian distribution with exponent s, where rank i is drawn
// with probability proportional to 1/i^s (cdf[i - 1] is the probability of drawing a rank of at most i).
static double *zipfian_cdf(double s, int max) {
    double *cdf = malloc(max * sizeof(double));
    double total = 0;
    for (int i = 1; i <= max; i++) {
        total += pow(i, -s);
        cdf[i - 1] = total;
    }
    for (int i = 0; i < max; i++) cdf[i] /= total;
    cdf[max - 1] = 1.0;
    return cdf;
}

// A rank from 1 to max drawn from the distribution of the given cumulative probabilities, by inverting them.
static int draw_zipfian(const double *cdf, int max) {
    double p = (next_random() >> 11) * 0x1.0p-53; // uniform in [0, 1)
    int lo = 0, hi = max - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (cdf[mid] > p) hi = mid;
        else lo = mid + 1;
    }
    return lo + 1;
}



int main(int argc, char **argv) {
    printf("Started writing queries\n");
    float zip_constant = 1.5f;
	if (argc < 4) {
		fprintf(stderr, "Please specify \nthe number of queries\nthe dataset\nthe max index\n(optional) the Zipfian constant");
		exit(1);
	} else if (argc > 4) {
        // zipfian constant if we want to do zipfian rather than uniform queries
        zip_constant = atof(argv[4]);
    }
    int num_queries = atoi(argv[1]);
    char *dataset = argv[2];
    int max = atoi(argv[3]);
    
    printf("Num queries: %i\n", num_queries);
    char output_name[100];
    sprintf(output_name, "../data/%s_%dM_%s.bin", ((argc > 4) ? "zipf" : "unif"), num_queries / 1000000, dataset);

    double *cdf = (argc > 4) ? zipfian_cdf(zip_constant, max) : NULL;
    FILE *f = fopen(output_name, "wb");
        for (int j = 0; j < num_queries; j++) {
            double value = 0;
            // If the user doesn't provide a zipfian constant, we assume we generate a uniform distribution instead
            if (argc > 4) {
                value = draw_zipfian(cdf, max);
            } else {
                value = rand_uniform(max);
            }
            int32_t result = (int32_t) value;
            fwrite(&result, sizeof(int32_t), 1, f);
        }
        fclose(f);
    free(cdf);
    return 0;
}