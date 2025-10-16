#include <stdio.h>
#include "include/rand_util.h"

double create_zipfian(double s, double max) {
    double p = (double)rand() / RAND_MAX;

    double pD = p * (12 * (pow(max, -s + 1) - 1) / (1 - s) + 6 + 6 * pow(max, -s) + s - s * pow(max, -s + 1));
    double x = max / 2;
    while (1) {
            double m = pow(x, -s - 2);
            double mx = m * x;
            double mxx = mx * x;
            double mxxx = mxx * x;

            double b = 12 * (mxxx - 1) / (1 - s) + 6 + 6 * mxx + s - (s * mx) - pD;
            double c = 12 * mxx - (6 * s * mx) + (m * s * (s + 1));
            double newx = x - b / c > 1 ? x - b / c : 1;
            if (abs(newx - x) <= 0.01) { // this is the tolerance for approximation
                    return newx;
            }
            x = newx;
    }
}

int main(int argc, char **argv) {
    printf("Started writing queries\n");
    float zip_constant = 1.5f;
	if (argc < 2) {
		fprintf(stderr, "Please specify \nthe number of queries\n(optional)the zipfian constant");
		exit(1);
	} else if (argc > 2) {
        // zipfian constant if we want to do zipfian rather than uniform queries
        zip_constant = atof(argv[2]);
    }
    int num_queries = atoi(argv[1]);
    printf("Num queries: %i\n", num_queries);

    int num_datasets = 3;
    char *datasets[] = {"news.bin", "url.bin", "ember.bin"};
    int max[] = {35919, 162798, 800000};
    for (int i = 0; i < num_datasets; i++) {
        FILE *f = fopen(datasets[i], "wb");
        for (int j = 0; j < num_queries; j++) {
            double value = 0;
            // If the user doesn't provide a zipfian constant, we assume we generate a uniform distribution instead
            if (argc > 2) {
                value = create_zipfian(zip_constant, max[i]);
            } else {
                value = rand_uniform((int)max[i]);
            }
            int32_t result = (int32_t) value;
            fwrite(&result, sizeof(int32_t), 1, f);
        }
        fclose(f);
    }
    return 0;
}