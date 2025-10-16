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
struct _set_node {
        struct _set_node *next;
        uint64_t key;
	uint64_t value;
} typedef set_node;

int set_insert(set_node *set, int set_len, uint64_t key, uint64_t value);

int set_query(set_node *set, int set_len, uint64_t key, uint64_t *value);

int set_delete(set_node *set, int set_len, uint64_t key);

int set_free(set_node *set, int set_len);

int insert_key(QF *qf, set_node *set, uint64_t set_len, uint64_t key, int count, struct timeval *timecheck, uint64_t *filter_time, uint64_t *set_time);

int get_obj_index(char *filename);

int get_label_index(char *filename);

char* get_dataset_name(char *filename);

char* get_dist_name(char *filename);

int split_csv_fields(char *line, char *fields[], int max_fields);

#endif