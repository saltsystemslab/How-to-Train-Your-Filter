#ifndef _TEST_DRIVER_H_
#define _TEST_DRIVER_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <openssl/rand.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"
#include "include/hashutil.h"
#include "include/rand_util.h"

#define HASH_SET_SEED 26571997

// Hash map node definition
struct _set_node {
        struct _set_node *next;
        uint64_t key;
	uint64_t value;
} typedef set_node;

// Struct to hold key and count pairs
struct key_count {
    uint64_t key;
    uint64_t count;
};

// Filter + Reverse Map Operations -----------------------------------

// Inserts an element into the hash set reverse map
int set_insert(set_node *set, int set_len, uint64_t key, uint64_t value);

// Updates the value of an element in the hash set reverse map
int set_update(set_node *set, int set_len, uint64_t key, uint64_t value);

// Checks if an element exists in the hash set reverse map
int set_query(set_node *set, int set_len, uint64_t key, uint64_t *value);

// Removes an element from the hash set reverse map
int set_delete(set_node *set, int set_len, uint64_t key);

// Frees the memory allocated for the hash table reverse map
int set_free(set_node *set, int set_len);

// Inserts a key into both the hash table reverse map and the AQF
int insert_key(QF *qf, set_node *set, uint64_t set_len, uint64_t key, int count, struct timeval *timecheck, uint64_t *filter_time, uint64_t *set_time);

// Returns the size, in bytes, of the given AQF
uint64_t get_aqf_size(QF *qf);

// Returns a new allocated AQF according to the given build parameters
QF build_aqf(uint64_t nslots, uint64_t nhashbits);

// Used for parsing files --------------------------------------------

// For the given file/dataset, returns the column containing the key to store
int get_obj_index(char *filename);

// For the given file/dataset, returns the column containing the label/classification of the element
int get_label_index(char *filename);

// For the given (data) file, returns a file name to use for convenient output
char* get_dataset_name(char *filename);

// For the given (query) file, returns a convenient name describing the distribution the queries follow
char* get_dist_name(char *filename);

// For the given file, reads the data into arrays of file offsets (per row), insertion elements, and the number of inserts
int read_file(char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *insert_set, int *num_inserts);

// For the given file, reads the data into arrays of file offsets (per row) and query elements and query labels, the number of positive elements, and the number of queries.
int read_queries(char *indexfilename, char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *query_set, uint64_t *query_labels, int *pos_count, int *query_count);

// Utility functions ------------------------------------------------

// define descending order comparison for qsort
int key_count_comp(const void *a, const void *b);

// hashes a given string
uint64_t hash_str(char *str);
#endif