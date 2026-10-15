#include <stdio.h>
#include <sys/stat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <openssl/rand.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"
#include "include/hashutil.h"
#include "include/rand_util.h"

#include <sys/file.h>
#include "include/exp_utility.h"

#define READ_SIZE 4096
#define URL_SIZE 1024
#define URL_PATH "../data/malicious_url_scores.csv"
#define EMBER_PATH "../data/combined_ember_metadata.csv"
#define SHALLA_PATH "../data/shalla_combined.csv"
#define CAIDA_PATH "../data/caida.csv"

// Updates an element in the hash set reverse map.
int set_update(set_node *set, int set_len, uint64_t key, uint64_t value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->occupied) {
		return 0;
	}
	else {
		while (ptr) {
			if (ptr->key == key) {
				ptr->value = value;
				return 1;
			}
			ptr = ptr->next;
		}
		return 0;
	}
}

// Inserts an element into the hash set reverse map.
int set_insert(set_node *set, int set_len, uint64_t key, uint64_t value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->occupied) {
		ptr->key = key;
		ptr->value = value;
		ptr->occupied = 1;
    }
	else {
		// if the slot is taken, do a chained insert
		while (ptr) {
			if (ptr->key == key) return 0; // key already exists
			if(!ptr->next) break;
			ptr = ptr->next;
		}
		set_node *node = malloc(sizeof(set_node));
		ptr->next = node;

		node->next = NULL;
		node->key = key;
		node->value = value;
		node->occupied = 1;
	}
	return 1;
}

// Checks if an element exists in the hash set reverse map.
int set_query(set_node *set, int set_len, uint64_t key, uint64_t *value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->occupied) {
		return 0;
	}
	else {
		if (ptr->key == key){
			*value = ptr->value;
			return 1;
		}
		while (ptr) {
			if (ptr->key == key) {
				*value = ptr->value;
				return 1;
			}
			ptr = ptr->next;
		}
		return 0;
	}
}

// Removes an element from the hash table reverse map.
int set_delete(set_node *set, int set_len, uint64_t key) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->occupied) {
        // fprintf(stderr, "nothing to delete\n");
		return 0;
	}
	else if (ptr->key == key) {
		if (ptr->next) {
			set_node *temp = ptr->next;
			ptr->key = ptr->next->key;
			ptr->value = ptr->next->value;
			ptr->next = ptr->next->next;
			ptr->occupied = 1;
			free(temp);
		}
		else {
			ptr->key = 0;
			ptr->value = 0;
			ptr->occupied = 0;
		}
		return 1;
	}
	else if (!ptr->next) {
		return 0;
	}
	else {	
		do {
			if (ptr->next->key == key) {
				set_node *temp = ptr->next;
				ptr->next = ptr->next->next;
				free(temp);
				return 1;
			}
			ptr = ptr->next;
		} while (ptr->next);
		return 0;
	}
}

// Frees the memory allocated for the hash table reverse map.
int set_free(set_node *set, int set_len) {
	for (int i = 0; i < set_len; i++) {
		set_node *ptr = &set[i];
		if (!ptr->occupied) continue;
		while (ptr->next) {
			set_node *temp = ptr->next;
			ptr->next = ptr->next->next;
			free(temp);
		}
	}
	free(set);
	return 1;
}

// Inserts a key into both the hash table reverse map and the AQF.
int insert_key(QF *qf, set_node *set, uint64_t set_len, uint64_t key, int count, struct timeval *timecheck, uint64_t *filter_time, uint64_t *set_time) {
    uint64_t ret_index, ret_hash, ret_other_hash;
	uint64_t start_time, end_time;
	uint64_t total_filter_time = 0;
	uint64_t total_set_time = 0;
	int ret_hash_len;
	start_time = TIMED_NOW();
	int ret = qf_insert_ret(qf, key, count, &ret_index, &ret_hash, &ret_hash_len, QF_NO_LOCK | QF_KEY_IS_HASH);
	ret_hash += (ret_hash == 0) ? 1 : 0; // TODO : this is super sus!!!
	end_time = TIMED_NOW();
	total_filter_time += end_time - start_time;
	if (ret == QF_NO_SPACE) {
		return 0;
	}
	else if (ret == 0) {
		// insert didn't occur because the key was already present in the filter (lines 1229-1237 gqf.c)
		uint64_t orig_key;
		start_time = TIMED_NOW();
		ret = set_query(set, set_len, ret_hash, &orig_key);
		end_time = TIMED_NOW();
		total_set_time += end_time - start_time;
		if (!ret) {
			fprintf(stderr, "error:\tfilter claimed to have fingerprint %lu but hashtable could not find it\n", ret_hash);
			if (ret_hash == 0) {
				// rare case where in an empty filter the empty slot is misidentified
				start_time = TIMED_NOW();
				set_insert(set, set_len, ret_hash, key);
				end_time = TIMED_NOW();
				total_set_time += end_time - start_time;
				*filter_time = total_filter_time;
				*set_time = total_set_time;
				return 1;
			}
			return 0;
        }
		if (key != orig_key) {
			// If the two values are different, there is a hash collision and we need to extend both fingerprints.
			// We reinsert by deleting the old fingerprint and inserting both new ones.
			uint64_t old_hash = ret_hash;
			start_time = TIMED_NOW();
			int ext_len = insert_and_extend(qf, ret_index, key, count, orig_key, &ret_hash, &ret_other_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
			end_time = TIMED_NOW();
			total_filter_time += end_time - start_time;
			if (ext_len == QF_NO_SPACE) {
				printf("filter is full after insert_and_extend\n");
				return 0;
			}
			start_time = TIMED_NOW();
			set_delete(set, set_len, old_hash);
			set_insert(set, set_len, ret_other_hash, orig_key);
			set_insert(set, set_len, ret_hash, key);
			end_time = TIMED_NOW();
			total_set_time += end_time - start_time;
		}
		*filter_time = total_filter_time;
		*set_time = total_set_time;
		return 1;
	}
	else if (ret == 1) {
		// insert successfully occurred, so just add the new element to the set
		start_time = TIMED_NOW();
		set_insert(set, set_len, ret_hash, key);
		end_time = TIMED_NOW();
		total_set_time += end_time - start_time;
		*filter_time = total_filter_time;
		*set_time = total_set_time;
		return 1;
	}
	printf("other error: errno %d\n", ret);
	return 0;
}

// For the given file, returns the column containing the elements to store.
int get_obj_index(char *filename) {
	if (strcmp(filename, URL_PATH) == 0) {
		return 0;
	} else if (strcmp(filename, EMBER_PATH) == 0) {
		return 2;
	} else if (strcmp(filename, CAIDA_PATH) == 0) {
		return 0;
	} else if (strcmp(filename, SHALLA_PATH) == 0) {
		return 0;
	} else {
		fprintf(stderr, "No filename specified\n");
		exit(1);
	}
}

// For the given file, returns the column containing the labels of the elements.
int get_label_index(char *filename) {
	if (strcmp(filename, URL_PATH) == 0) {
		return 1;
	} else if (strcmp(filename, EMBER_PATH) == 0) {
		return 4;
	} else if (strcmp(filename, CAIDA_PATH) == 0) {
		return 4;
	} else if (strcmp(filename, SHALLA_PATH) == 0) {
		return 1;
	} else {
		fprintf(stderr, "No filename specified\n");
		exit(1);
	}
}

// For the given (data) file, returns a file name to use for convenient output.
char* get_dataset_name(char *filename) {
	char* datasets[] = {"url", "ember", "caida", "shalla"};	
	for (int i = 0; i < 4; i++) {
		if (strstr(filename, datasets[i]) != NULL) {
			return datasets[i];
		}
	}
	return NULL;
}

// For the given (query) file, returns a convenient name describing the distribution that the queries follow.
char* get_dist_name(char *filename) {
	if (strstr(filename, "unif") != NULL) {
		return "unif";
	} else if (strstr(filename, "zipf") != NULL) {
		return "zipf";
	} else {
		return "other";
	}
}

// For the given file, reads the data into arrays of file offsets (per row) and insertion elements, and the number of inserts.
int read_file(char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *insert_set, int *num_inserts, int *insert_rows) {
	FILE * file_ptr;
	file_ptr = fopen(filename, "r");
	if (NULL == file_ptr) {
		printf("file can't be opened \n");
		return 0;
	}
	int curr_inserts = 0;
	int curr_index = 0;
	fgets(buffer, READ_SIZE, file_ptr); // get rid of the first header row
	offsets[curr_index] = ftell(file_ptr);
	while (fgets(buffer, READ_SIZE, file_ptr)) {
		if (strlen(buffer) <= 1) {
			continue; // skip blank lines
		}
		if (strcmp(filename, URL_PATH) == 0) {
			if (strstr(buffer, ",malicious,")) {
				char *url;
				char url_buf[URL_SIZE];
				if (buffer[0] == '"') {
					int url_index = 1;
					while (buffer[url_index] != '"') url_index++;
					strncpy(url_buf, buffer + 1, url_index-1);
					url_buf[url_index-1] = '\0';
					url = url_buf;
				} else {
					// otherwise, simply dividing the str with "," gives the url
					url = strtok(buffer, ",");
				}
				if (url != NULL && strlen(url) > 0) {
					if (insert_rows) insert_rows[curr_inserts] = curr_index;
					insert_set[curr_inserts++] = hash_str(url);
				}
			} 
		} else {
				// normally formatted with 1 as a positive indicator
				// read label and item then insert as necessary
				char *label = NULL;
				char *item = NULL;
				char *token = strtok(buffer, ",");
				int count = 0;
				while (token != NULL) {
					if (count == label_index) {
						label = token;
					} else if (count == obj_index) {
						item = token;
					}
					token = strtok(NULL, ",");
					count++;
				}
				if (label != NULL && atoi(label) == 1 && item != NULL) {
					if (insert_rows) insert_rows[curr_inserts] = curr_index;
					insert_set[curr_inserts++] = hash_str(item);
				}
			}
			curr_index++;
			offsets[curr_index] = ftell(file_ptr);
	}
	*num_inserts = curr_inserts;
	fclose(file_ptr);
	return 1;
}

// For the given file, reads the data into arrays of file offsets (per row) and query elements and query labels, the number of positive elements, and the number of queries.
int read_queries(char *indexfilename, char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *query_set, uint64_t *query_labels, size_t max_queries, int *pos_count, int *query_count, int *query_rows) {
	FILE * index_file_ptr = fopen(indexfilename, "r");
	if (!index_file_ptr) {
		printf("couldn't open index file\n");
		return 0;
	}
	FILE * file_ptr = fopen(filename, "r");
	if (!file_ptr) {
		printf("couldn't open index file\n");
		return 0;
	}
	fgets(buffer, READ_SIZE, index_file_ptr); // get rid of the header row
	int curr_query = 0;
	int num_pos = 0;
	// stop once the query arrays are full, even if the index file has more rows
	while ((size_t)curr_query < max_queries && fgets(buffer, READ_SIZE, index_file_ptr)) {
		// obtain the index from the current row
		int index = atoi(strtok(buffer, ","));
		if (query_rows) query_rows[curr_query] = index;
		// use offsets array to find element in original data
		if (fseek(file_ptr, offsets[index], SEEK_SET) != 0) {
			fprintf(stderr, "fseek failed");
			fclose(index_file_ptr);
			return EXIT_FAILURE;
		}
		// obtain the object we want to insert...
		if (!fgets(buffer, READ_SIZE, file_ptr)) {
			fprintf(stderr, "failed to read from file\n");
		}
		if (strcmp(filename, URL_PATH) == 0) {
			if (strstr(buffer, ",malicious,")) {
				query_labels[curr_query] = 1;
				num_pos++;
			} else {
				query_labels[curr_query] = 0;
			}
			char *url;
			char url_buf[URL_SIZE];
			if (buffer[0] == '"') {
				int url_index = 1;
				while (buffer[url_index] != '"') url_index++;
				strncpy(url_buf, buffer + 1, url_index-1);
				url_buf[url_index-1] = '\0';
				url = url_buf;
			} else {
				url = strtok(buffer, ",");
			}
			query_set[curr_query] = hash_str(url);
		} else {
			char *label = NULL;
			char *item = NULL;
			char *token = strtok(buffer, ",");
			int count = 0;
			while (token != NULL) {
				if (count == label_index) {
					label = token;
				} else if (count == obj_index) {
					item = token;
				}
				token = strtok(NULL, ",");
				count++;
			}
			if (item == NULL) {
				fprintf(stderr, "skipping query row %d: no key column\n", index);
				continue;
			}
			// insert all corresponding indexes into the query set.
			query_set[curr_query] = hash_str(item);
			// also track the corresponding label for the query...
			
			if (label != NULL) {
				query_labels[curr_query] = atoi(label);
				if (query_labels[curr_query] == 1) {
					num_pos++;
				}
			}
		}
		curr_query++;
	}
	*pos_count = num_pos;
	*query_count = curr_query;
	fclose(index_file_ptr);
	fclose(file_ptr);
	return 1;
}

// returns the size of the given AQF
uint64_t get_aqf_size(QF *qf) {
	return qf->metadata->total_size_in_bytes;
}

// returns an AQF according to the given build parameters
QF build_aqf(uint64_t nslots, uint64_t nhashbits) {
	QF qf;
	if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {
		fprintf(stderr, "Can't allocate AQF.\n");
		abort();
	}
	qf_set_auto_resize(&qf, false);
	return qf;
}

// define descending order comparison for qsort
int key_count_comp(const void *a, const void *b) {
    uint64_t val_a = ((struct key_count *)a)->count;
    uint64_t val_b = ((struct key_count *)b)->count;
    return (val_a < val_b) - (val_a > val_b);
}

// hashes a given string to the 64-bit key the filters store. (This was djb2, whose collisions between short strings,
// e.g. "gvt1.com" and "gx21.com", made some negative keys of the shalla dataset identical to positive keys.)
#define HASH_STR_SEED 0x2c1b3c6d
uint64_t hash_str(char *str) {
	return MurmurHash64A(str, (int)strlen(str), HASH_STR_SEED);
}

// For the given file, reads positives into insert_set and negatives into negative_set, along with file offsets (per row).
int read_file_with_negatives(char *filename, int obj_index, int label_index, char *buffer, long *offsets,
							 uint64_t *insert_set, int *num_inserts, uint64_t *negative_set, int *num_negatives,
							 int *insert_rows, int *negative_rows) {
	FILE * file_ptr = fopen(filename, "r");
	if (NULL == file_ptr) {
		printf("file can't be opened \n");
		return 0;
	}
	int curr_inserts = 0;
	int curr_neg = 0;
	int curr_index = 0;
	fgets(buffer, READ_SIZE, file_ptr); // get rid of the first header row
	offsets[curr_index] = ftell(file_ptr);
	while (fgets(buffer, READ_SIZE, file_ptr)) {
		if (strlen(buffer) == 0) {
			continue; // skip blank lines
		}
		if (strcmp(filename, URL_PATH) == 0) {
			char *malicious = strstr(buffer, ",malicious,");
			char *url;
			char url_buf[READ_SIZE];
			if (buffer[0] == '"') {
				// if the first character is a ", it means that the element is a weirdly-formatted url
				int url_index = 1;
				while (buffer[url_index] != '"') url_index++;
				strncpy(url_buf, buffer + 1, url_index-1);
				url_buf[url_index-1] = '\0';
				url = url_buf;
			} else {
				// otherwise, simply dividing the str with "," gives the url
				url = strtok(buffer, ",");
			}
			if (url != NULL) {
				if (malicious) {
					if (insert_rows) insert_rows[curr_inserts] = curr_index;
					insert_set[curr_inserts++] = hash_str(url);
				} else {
					if (negative_rows) negative_rows[curr_neg] = curr_index;
					negative_set[curr_neg++] = hash_str(url);
				}
			}
		} else {
			// normally formatted with 1 as a positive indicator
			char *label = NULL;
			char *item = NULL;
			char *token = strtok(buffer, ",");
			int count = 0;
			while (token != NULL) {
				if (count == label_index) {
					label = token;
				} else if (count == obj_index) {
					item = token;
				}
				token = strtok(NULL, ",");
				count++;
			}
			if (item == NULL) {
				// no key column on this row
			} else if (label != NULL && atoi(label) == 1) {
				if (insert_rows) insert_rows[curr_inserts] = curr_index;
				insert_set[curr_inserts++] = hash_str(item);
			} else {
				if (negative_rows) negative_rows[curr_neg] = curr_index;
				negative_set[curr_neg++] = hash_str(item);
			}
		}
		curr_index++;
		offsets[curr_index] = ftell(file_ptr);
	}
	fclose(file_ptr);
	*num_inserts = curr_inserts;
	*num_negatives = curr_neg;
	return 1;
}

// Collects the unique negative keys in the first num_check queries, sorted by descending frequency, along with their
// cumulative frequency distribution. Same logic as the stacked filter experiments.
int build_training_negatives(uint64_t *query_set, uint64_t *query_labels, size_t num_queries, int num_check,
							 uint64_t **sorted_negatives_out, double **cdf_out) {
	// create a map of key to count, which we later convert to an array for sorting.
	set_node *count_map = calloc(num_queries, sizeof(set_node));
	if (!count_map) {
		fprintf(stderr, "calloc failed for count_map\n");
		exit(EXIT_FAILURE);
	}
	uint64_t value;
	int num_negative = 0;
	for (int i = 0; i < num_check; i++) {
		if (query_labels[i] == 0) {
			if (set_query(count_map, num_queries, query_set[i], &value)) {
				// key already exists, so just increment count
				set_update(count_map, num_queries, query_set[i], value + 1);
			} else {
				// key doesn't exist, start a new count
				set_insert(count_map, num_queries, query_set[i], 1);
				num_negative++;
			}
		}
	}
	fprintf(stderr, "found %d unique negative queries in %d training queries\n", num_negative, num_check);

	// now, convert the map to an array for sorting
	struct key_count *count_array = malloc(num_negative * sizeof(struct key_count));
	if (!count_array) {
		fprintf(stderr, "malloc failed for count_array\n");
		exit(EXIT_FAILURE);
	}
	int index = 0;
	for (size_t i = 0; i < num_queries; i++) {
		set_node *ptr = &count_map[i];
		while (ptr) {
			if (ptr->occupied) {
				count_array[index].key = ptr->key;
				count_array[index].count = ptr->value;
				index++;
			}
			ptr = ptr->next;
		}
	}
	set_free(count_map, num_queries); // also frees count_map itself

	// now, sort the array in descending order by count
	qsort(count_array, num_negative, sizeof(struct key_count), key_count_comp);

	uint64_t *sorted_negatives = malloc(num_negative * sizeof(uint64_t));
	double *cdf = malloc(num_negative * sizeof(double));
	if (!sorted_negatives || !cdf) {
		fprintf(stderr, "malloc failed for sorted_negatives or cdf\n");
		exit(EXIT_FAILURE);
	}
	uint64_t total_count = 0;
	for (int i = 0; i < num_negative; i++) {
		sorted_negatives[i] = count_array[i].key;
		total_count += count_array[i].count;
	}
	double sum = 0;
	for (int i = 0; i < num_negative; i++) {
		sum += (double)count_array[i].count / (double)total_count;
		cdf[i] = sum;
	}
	free(count_array);

	*sorted_negatives_out = sorted_negatives;
	*cdf_out = cdf;
	return num_negative;
}

// Loads a matrix stored in CascadedLBF's binary format (size_t rows, size_t cols, then row-major float32).
float *load_float_matrix(const char *path, size_t *rows, size_t *cols) {
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "couldn't open %s\n", path);
		return NULL;
	}
	uint64_t header[2];
	if (fread(header, sizeof(uint64_t), 2, f) != 2) {
		fprintf(stderr, "couldn't read the header of %s\n", path);
		fclose(f);
		return NULL;
	}
	float *data = malloc(header[0] * header[1] * sizeof(float));
	if (!data || fread(data, sizeof(float), header[0] * header[1], f) != header[0] * header[1]) {
		fprintf(stderr, "couldn't read the data of %s\n", path);
		free(data);
		fclose(f);
		return NULL;
	}
	fclose(f);
	*rows = header[0];
	*cols = header[1];
	return data;
}

float *map_float_matrix(const char *path, size_t *rows, size_t *cols) {
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "couldn't open %s\n", path);
		return NULL;
	}
	struct stat st;
	uint64_t header[2];
	if (fstat(fd, &st) != 0 || pread(fd, header, sizeof(header), 0) != sizeof(header)
		|| (uint64_t)st.st_size != sizeof(header) + header[0] * header[1] * sizeof(float)) {
		fprintf(stderr, "couldn't read the header of %s\n", path);
		close(fd);
		return NULL;
	}
	char *base = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	close(fd);
	if (base == MAP_FAILED) {
		fprintf(stderr, "couldn't map %s\n", path);
		return NULL;
	}
	*rows = header[0];
	*cols = header[1];
	return (float *)(base + sizeof(header));
}

void unmap_float_matrix(float *data, size_t rows, size_t cols) {
	if (data) munmap((char *)data - 2 * sizeof(uint64_t), 2 * sizeof(uint64_t) + rows * cols * sizeof(float));
}

// Copies the given rows of a row-major matrix with num_cols columns into out (num_selected * num_cols floats).
void gather_rows(const float *matrix, size_t num_cols, const int *rows, size_t num_selected, float *out) {
	for (size_t i = 0; i < num_selected; i++) {
		memcpy(out + i * num_cols, matrix + (size_t)rows[i] * num_cols, num_cols * sizeof(float));
	}
}

// Checks that the harness's view of each row matches the exported labels (row_labels, one per dataset row),
// which catches any misalignment between the harness's row indices and the exported features.
// labels holds the expected label of each of the num_selected rows, or NULL if they are all positives.
int check_row_labels(const float *row_labels, size_t num_rows, const int *rows, const uint64_t *labels, size_t num_selected, const char *what) {
	for (size_t i = 0; i < num_selected; i++) {
		uint64_t expected = labels ? labels[i] : 1;
		if (rows[i] < 0 || (size_t)rows[i] >= num_rows || (uint64_t)row_labels[rows[i]] != expected) {
			fprintf(stderr, "%s %zu (row %d) has label %lu in the harness but not in the exported labels; the features do not line up with the dataset\n",
					what, i, rows[i], expected);
			return 0;
		}
	}
	return 1;
}

// Opens a results file for appending, positioned at its end, holding an exclusive lock until it is closed, so that
// experiments writing the same file at the same time cannot interleave their rows or both write the header.
FILE *open_results_file(const char *path) {
	FILE *f = fopen(path, "a");
	if (!f) {
		fprintf(stderr, "couldn't open results file %s\n", path);
		exit(EXIT_FAILURE);
	}
	flock(fileno(f), LOCK_EX);
	fseek(f, 0, SEEK_END);
	return f;
}
