/*
Formatting of the measurement columns shared by the distribution, one-pass and adversarial results.

All times are in nanoseconds. Builds with TRACK_FP_SOURCE do not time anything (their timing columns are
empty) and instead fill in fp_from_model and fp_from_backup for filters that report where their false
positives came from; other builds leave those two columns empty.
*/
#ifndef HARNESS_OUTPUT_H
#define HARNESS_OUTPUT_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "include/filter_iface.h"

#define MEASUREMENT_COLUMNS "build_time_ns,aux_build_time_ns,build_model_ns,build_filter_ns," \
	"amortized_query_ns,query_model_ns,query_filter_ns,amortized_adapt_ns,expected_fpr,config_time_ns," \
	"fp_from_model,fp_from_backup,model_size,num_adaptations"

// Writes ",<model_size>,<num_adaptations>": the size of the filter's model (0 without a model) and its
// successful adaptations (empty for filters that do not adapt). Recorded in every build.
static inline void format_model_and_adaptations(char *buf, size_t len, const filter_ops *ops, void *filter, uint64_t extra_adaptations) {
	char adaptations[32] = "";
	if (ops->num_adaptations) snprintf(adaptations, sizeof(adaptations), "%lu", ops->num_adaptations(filter) + extra_adaptations);
	snprintf(buf, len, ",%zu,%s", ops->model_size_bytes ? ops->model_size_bytes(filter) : 0, adaptations);
}

// build_time: building the filter from its configured parameters (excluding config_time and sizing_time);
// aux_time: the part of build_time spent on auxiliary structures (e.g. the AQF's reverse map).
static inline void format_measurements(char *buf, size_t len, const filter_ops *ops, void *filter,
									   uint64_t build_time, uint64_t aux_time, uint64_t total_query_time,
									   uint64_t total_adapt_time, size_t num_queries, uint64_t config_time,
									   uint64_t fp_from_model, uint64_t fp_from_backup) {
	char expected_fpr[32] = "";
	if (ops->expected_fpr) snprintf(expected_fpr, sizeof(expected_fpr), "%.10g", ops->expected_fpr(filter));
#ifdef TRACK_FP_SOURCE
	(void)build_time; (void)aux_time; (void)total_query_time; (void)total_adapt_time; (void)num_queries; (void)config_time;
	if (ops->last_query_source) {
		snprintf(buf, len, ",,,,,,,,%s,,%lu,%lu", expected_fpr, fp_from_model, fp_from_backup);
	} else {
		snprintf(buf, len, ",,,,,,,,%s,,,", expected_fpr);
	}
	size_t used = strlen(buf);
	format_model_and_adaptations(buf + used, len - used, ops, filter, 0);
#else
	(void)fp_from_model; (void)fp_from_backup;
	double amortized_query = (double)total_query_time / num_queries;
	// filters without a model spend all their time in the filter itself
	filter_timing t = {0, build_time - aux_time, 0, 0, 0};
	double query_model = 0, query_filter = amortized_query;
	if (ops->timing) {
		ops->timing(filter, &t);
		query_model = t.timed_queries ? (double)t.query_model_ns / t.timed_queries : 0;
		query_filter = t.timed_queries ? (double)t.query_filter_ns / t.timed_queries : 0;
	}
	snprintf(buf, len, "%lu,%lu,%lu,%lu,%.3f,%.3f,%.3f,%.3f,%s,%lu,,",
			 build_time, aux_time, t.build_model_ns, t.build_filter_ns, amortized_query, query_model, query_filter,
			 (double)total_adapt_time / num_queries, expected_fpr, config_time);
	size_t used = strlen(buf);
	format_model_and_adaptations(buf + used, len - used, ops, filter, 0);
#endif
}

// Counts a false positive by where it came from, for filters that report it (TRACK_FP_SOURCE builds only).
static inline void count_fp_source(const filter_ops *ops, void *filter, uint64_t *fp_from_model, uint64_t *fp_from_backup) {
#ifdef TRACK_FP_SOURCE
	if (ops->last_query_source) {
		if (ops->last_query_source(filter) == FP_SOURCE_MODEL) {
			(*fp_from_model)++;
		} else {
			(*fp_from_backup)++;
		}
	}
#else
	(void)ops; (void)filter; (void)fp_from_model; (void)fp_from_backup;
#endif
}

#endif
