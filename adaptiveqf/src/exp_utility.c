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
#include <pthread.h>
#include <openssl/rand.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"
#include "include/hashutil.h"
#include "include/rand_util.h"

#include "include/exp_utility.h"

#define READ_SIZE 4096
#define URL_PATH "../data/malicious_url_scores.csv"
#define EMBER_PATH "../data/combined_ember_metadata.csv"
#define SHALLA_PATH "../data/shalla_combined.csv"
#define CAIDA_PATH "../data/caida.csv"

// Inserts an element into the hash set reverse map.
int set_insert(set_node *set, int set_len, uint64_t key, uint64_t value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->key) {
		ptr->key = key;
		ptr->value = value;
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
	}
	return 1;
}

// Checks if an element exists in the hash set reverse map.
int set_query(set_node *set, int set_len, uint64_t key, uint64_t *value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->key) {
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
	if (!ptr->key) {
        // fprintf(stderr, "nothing to delete\n");
		return 0;
	}
	else if (ptr->key == key) {
		if (ptr->next) {
			set_node *temp = ptr->next;
			ptr->key = ptr->next->key;
			ptr->value = ptr->next->value;
			ptr->next = ptr->next->next;
			free(temp);
		}
		else {
			ptr->key = 0;
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
		if (!ptr->key) continue;
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
	gettimeofday(timecheck, NULL);
	start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
	int ret = qf_insert_ret(qf, key, count, &ret_index, &ret_hash, &ret_hash_len, QF_NO_LOCK | QF_KEY_IS_HASH);
	gettimeofday(timecheck, NULL);
	end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
	total_filter_time += end_time - start_time;
	if (ret == QF_NO_SPACE) {
		return 0;
	}
	else if (ret == 0) {
		// insert didn't occur because the key was already present in the filter (lines 1229-1237 gqf.c)
		uint64_t fingerprint = ret_hash | (1ull << ret_hash_len), orig_key;
		gettimeofday(timecheck, NULL);
		start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		ret = set_query(set, set_len, ret_hash, &orig_key);
		gettimeofday(timecheck, NULL);
		end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		total_set_time += end_time - start_time;
		if (!ret) {
			printf("error:\tfilter claimed to have fingerprint %lu but hashtable could not find it\n", ret_hash);
			return 0;
        }
		if (key != orig_key) {
			// If the two values are different, there is a hash collision and we need to extend both fingerprints.
			// We reinsert by deleting the old fingerprint and inserting both new ones.
			uint64_t old_hash = ret_hash;
			gettimeofday(timecheck, NULL);
			start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
			int ext_len = insert_and_extend(qf, ret_index, key, count, orig_key, &ret_hash, &ret_other_hash, QF_KEY_IS_HASH | QF_NO_LOCK);
			gettimeofday(timecheck, NULL);
			end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
			total_filter_time += end_time - start_time;
			if (ext_len == QF_NO_SPACE) {
				printf("filter is full after insert_and_extend\n");
				return 0;
			}
			gettimeofday(timecheck, NULL);
			start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
			set_delete(set, set_len, old_hash);
			set_insert(set, set_len, ret_other_hash, orig_key);
			set_insert(set, set_len, ret_hash, key);
			gettimeofday(timecheck, NULL);
			end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
			total_set_time += end_time - start_time;
		}
		*filter_time = total_filter_time;
		*set_time = total_set_time;
		return 1;
	}
	else if (ret == 1) {
		// insert successfully occurred, so just add the new element to the set
		gettimeofday(timecheck, NULL);
		start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		set_insert(set, set_len, ret_hash, key);
		gettimeofday(timecheck, NULL);
		end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
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
int read_file(char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *insert_set, int *num_inserts) {
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
		if (strlen(buffer) == 0) {
			continue; // skip blank lines
		}
		if (strcmp(filename, URL_PATH) == 0) {
			if (strstr(buffer, ",malicious,")) {
				char *url;
				if (buffer[0] == "\"") {
					// if the first character is a ", it means that the element is a weirdly-formatted url
					int url_index = 1;
					while (buffer[url_index] != "\"") url_index++;
					// the url will be the substring from index 1 to url_index
					strncpy(url, buffer + 1, url_index-1);
				} else {
					// otherwise, simply dividing the str with "," gives the url
					url = strtok(buffer, ",");
				}
				if (url != NULL) {
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
				if (label != NULL && atoi(label) == 1) {
					char *item_copy = strdup(item);
					insert_set[curr_inserts++] = hash_str(item_copy);
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
int read_queries(char *indexfilename, char *filename, int obj_index, int label_index, char *buffer, long *offsets, uint64_t *query_set, uint64_t *query_labels, int *pos_count, int *query_count) {
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
	while (fgets(buffer, READ_SIZE, index_file_ptr)) {
		// obtain the index from the current row
		int index = atoi(strtok(buffer, ","));
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
			char *url = strtok(buffer, ",");
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
			// insert all corresponding indexes into the query set.
			char *item_copy = strdup(item);
			query_set[curr_query] = hash_str(item_copy);
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