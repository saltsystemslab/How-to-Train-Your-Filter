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

int set_insert(set_node *set, int set_len, uint64_t key, uint64_t value) {
	uint64_t hash = MurmurHash64A((void*)(&key), sizeof(key), HASH_SET_SEED);
	set_node *ptr = &set[hash % set_len];
	if (!ptr->key) {
		ptr->key = key;
		ptr->value = value;
    }
	else {
		// if the key doesn't exist, do a chained insert
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
        // fprintf(stderr, "no chains to check to delete\n");
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
        // fprintf(stderr, "no matching chains to delete\n");
		return 0;
	}
}

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
        // so, we need to check if it's present in the set and decide if we should extend or not

		uint64_t fingerprint = ret_hash | (1ull << ret_hash_len), orig_key;
		gettimeofday(timecheck, NULL);
		start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		ret = set_query(set, set_len, fingerprint, &orig_key);
		gettimeofday(timecheck, NULL);
		end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		total_set_time += end_time - start_time;
		if (!ret) {
			printf("error:\tfilter claimed to have fingerprint %lu but hashtable could not find it\n", ret_hash);
            // If the filter has the fingerprint but the set doesn't,
			// it means that there was some update that didn't make it to the set.
			// To keep it consistent, add the updated element to the set.
			// Note that this technically shouldn't happen.
            gettimeofday(timecheck, NULL);
            start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
            set_insert(set, set_len, ret_hash | (1ull << ret_hash_len), key);
            gettimeofday(timecheck, NULL);
            end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
            total_set_time += end_time - start_time;
			return 1;
        }
		if (ret) {
			// the filter already has the key, and it was also found in the set.
            // based on the value corresponding to the key, we decide how to extend the key(s).
			if (key == orig_key) {
				// if the two values are the same, just extend the existing key
				// by deleting the old fingerprint and inserting the new one.
				gettimeofday(timecheck, NULL);
				start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
				int ext_len = insert_and_extend(qf, ret_index, key, count, key, &ret_hash, &ret_other_hash, QF_NO_LOCK | QF_KEY_IS_HASH);
				if (ext_len == QF_NO_SPACE) {
					printf("filter is full after insert_and_extend\n");
					return 0;
				}
				set_delete(set, set_len, fingerprint);
				set_insert(set, set_len, ret_hash | (1ull << ret_hash_len), key);
				gettimeofday(timecheck, NULL);
				end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
				total_filter_time += end_time - start_time;
			}
			else {
				// If the two values are different, there is a hash collision and we need to extend both fingerprints.
				// We reinsert by deleting the old fingerprint and inserting both new ones.
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
				set_delete(set, set_len, fingerprint);
				set_insert(set, set_len, ret_other_hash | (1ull << ext_len), orig_key);
				set_insert(set, set_len, ret_hash | (1ull << ext_len), key);
				gettimeofday(timecheck, NULL);
				end_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
				total_set_time += end_time - start_time;
			}
		}
		*filter_time = total_filter_time;
		*set_time = total_set_time;
		return 1;
	}
	else if (ret == 1) {
		// insert successfully occurred and it wasn't in the filter yet, so just add the new element to the set
		gettimeofday(timecheck, NULL);
		start_time = timecheck->tv_sec * 1000000 + timecheck->tv_usec;
		set_insert(set, set_len, ret_hash | (1ull << ret_hash_len), key); // fingerprint is key, key becomes value
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

int get_obj_index(char *filename) {
	if (strcmp(filename,"../learned/data/fake_news_predictions.csv") == 0) {
		return 0;
	} else if (strcmp(filename, "../learned/data/malicious_url_scores.csv") == 0) {
		return 0;
	} else if (strcmp(filename, "../learned/data/combined_ember_metadata.csv") == 0) {
		return 2;
	} else if (strcmp(filename, "../learned/data/caida.csv") == 0) {
		return 0;
	} else if (strcmp(filename, "../learned/data/shalla_combined.csv") == 0) {
		return 0;
	} else {
		fprintf(stderr, "No filename specified\n");
		exit(1);
	}
}

int get_label_index(char *filename) {
	if (strcmp(filename,"../learned/data/fake_news_predictions.csv") == 0) {
		return 1;
	} else if (strcmp(filename, "../learned/data/malicious_url_scores.csv") == 0) {
		return 1;
	} else if (strcmp(filename, "../learned/data/combined_ember_metadata.csv") == 0) {
		return 4;
	} else if (strcmp(filename, "../learned/data/caida.csv") == 0) {
		return 4;
	} else if (strcmp(filename, "../learned/data/shalla_combined.csv") == 0) {
		return 1;
	} else {
		fprintf(stderr, "No filename specified\n");
		exit(1);
	}
}

char* get_dataset_name(char *filename) {
	if (strstr(filename, "url") != NULL) {
		return "url";
	} else if (strstr(filename, "ember") != NULL) {
		return "ember";
	} else if (strstr(filename, "news") != NULL) {
		return "news";
	} else if (strstr(filename, "caida") != NULL) {
		return "caida";
	} else if (strstr(filename, "shalla") != NULL) {
		return "shalla";
	} else {
		return "other";
	}
}

char* get_dist_name(char *filename) {
	if (strstr(filename, "unif") != NULL) {
		return "unif";
	} else if (strstr(filename, "zipf") != NULL) {
		return "zipf";
	} else {
		return "other";
	}
}