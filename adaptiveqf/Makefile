CTARGETS=test_distribution_generic test_one_pass_generic test_advers_dist_generic test_dynamic_generic test_shifting_generic write_queries test_micro test_throughput test_split_throughput test_adversarial test_splinter_lltable_throughput test_splinter_parallel test_parallel# test_fill_varied_throughput test_near_full test_deletions test_merge test_hash_accesses test_bulk test_whitelist test_resize test_micro_throughput test_micro_write test_micro_read test_lltable_throughput
CXXTARGETS=test_ext_throughput test_ext_inc_throughput test_zipf_throughput test_ext_churn taf
SPLTARGETS=test_splinter_ops test_splinter_inserts test_splinter_inserts_2 test_splinter_throughput test_splinter_zipfian_histogram test_splinter_adversarial
PYTARGET=libadaptive_qf.so
# test_progress

ifndef D
	DEBUG=
	OPT=-O3 -DNDEBUG
	SPLINTERPATH=external/splinterdb/build/release/lib
	#SPLINTERPATH=external/splinterdb/btree
else
	DEBUG=-g
	OPT=-O0
	SPLINTERPATH=external/splinterdb/build/debug/lib
	#SPLINTERPATH=external/splinterdb/btree
endif

ifdef NH
	ARCH=
else
	ARCH=-msse4.2 -D__SSE4_2_
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

WRAPPER=external/StackedFilters/wrapper
BLOOM_WRAPPER=external/bloom_wrapper
# the Bloom filter used by CascadedLBF (header-only)
FASTFILTER=../CascadedLBF/thirdparty/fastfilter_cpp/src
# the learned Bloom filters (PLBF, CLBF) from CascadedLBF, built as shared libraries (see lbf_wrapper.h)
CASCADED=../CascadedLBF
CASCADED_WRAPPER=external/cascaded_wrapper
XGBOOST_LIB=$(abspath $(CASCADED)/thirdparty/xgboost/lib)
LBF_CXXFLAGS = -std=c++17 -Wall $(DEBUG) $(OPT) $(ARCH) -m64 -fno-strict-aliasing -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
	-I$(CASCADED_WRAPPER) -I$(CASCADED)/src -I$(CASCADED)/thirdparty -I$(CASCADED)/thirdparty/xgboost/include -I$(FASTFILTER) -I$(FASTFILTER)/bloom
BLOOM=external/StackedFilters/include/BloomFilter
OPTIMIZATION=external/StackedFilters/include/Optimization
LOC_INCLUDE=include
LOC_SRC=src
LOC_TEST=test
OBJDIR=obj

CC = gcc -std=gnu11
CXX = g++ -std=c++11
LD= gcc -std=gnu11

CXXFLAGS = -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -Iinclude -Iexternal/StackedFilters/wrapper -I$(BLOOM_WRAPPER) -I$(CASCADED_WRAPPER) -Iexternal/StackedFilters/include/StackedFilter -Iexternal/StackedFilters/include/BloomFilter -Iexternal/StackedFilters/include/Optimization -Iexternal/splinterdb/include -DSPLINTERDB_PLATFORM_DIR=platform_linux -DSKIP_BOOL_DEF -D_GNU_SOURCE

LDFLAGS = $(DEBUG) $(PROFILE) $(OPT) -lpthread -lssl -lcrypto -lm -L$(SPLINTERPATH) -lsplinterdb -Wl,-rpath=$(SPLINTERPATH)
LDFLAGS += -lstdc++
# the learned filter wrapper libraries live in obj/ next to the binaries
LDFLAGS += -Wl,-rpath,'$$ORIGIN/obj'
#LDFLAGS += -L/usr/lib/ -lstxxl

#
# declaration of dependencies
#

# Variants of the generic experiment harnesses built with TRACK_FP_SOURCE: they record whether learned
# filters' false positives came from unchecked model decisions or backup filter errors, and do not time
# anything (see include/harness_output.h).
FPSOURCE_TARGETS=test_distribution_generic_fpsource test_one_pass_generic_fpsource test_advers_dist_generic_fpsource test_dynamic_generic_fpsource test_shifting_generic_fpsource
FPSOURCE_OBJDIR=$(OBJDIR)/fpsource

all: $(CTARGETS) $(FPSOURCE_TARGETS) #$(SPLTARGETS) #$(CXXTARGETS)

python: $(PYTARGET)

# dependencies between programs and .o files

write_queries:							$(OBJDIR)/write_queries.o $(OBJDIR)/rand_util.o

test_unit:								$(OBJDIR)/test_unit.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_throughput:						$(OBJDIR)/test_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_split_throughput:						$(OBJDIR)/test_split_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_adversarial:						$(OBJDIR)/test_adversarial.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_micro:								$(OBJDIR)/test_micro.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_distribution_generic:				$(OBJDIR)/test_distribution_generic.o $(OBJDIR)/filter_registry.o \
										$(OBJDIR)/aqf_adapter.o $(OBJDIR)/stacked_adapter.o $(OBJDIR)/stacked_wrapper.o \
										$(OBJDIR)/bloom_adapter.o $(OBJDIR)/bloom_wrapper.o \
										$(OBJDIR)/lbf_adapter.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
										$(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/exp_utility.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_one_pass_generic:				$(OBJDIR)/test_one_pass_generic.o $(OBJDIR)/filter_registry.o \
										$(OBJDIR)/aqf_adapter.o $(OBJDIR)/stacked_adapter.o $(OBJDIR)/stacked_wrapper.o \
										$(OBJDIR)/bloom_adapter.o $(OBJDIR)/bloom_wrapper.o \
										$(OBJDIR)/lbf_adapter.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
										$(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/exp_utility.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_advers_dist_generic:				$(OBJDIR)/test_advers_dist_generic.o $(OBJDIR)/filter_registry.o \
										$(OBJDIR)/aqf_adapter.o $(OBJDIR)/stacked_adapter.o $(OBJDIR)/stacked_wrapper.o \
										$(OBJDIR)/bloom_adapter.o $(OBJDIR)/bloom_wrapper.o \
										$(OBJDIR)/lbf_adapter.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
										$(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/exp_utility.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_dynamic_generic:				$(OBJDIR)/test_dynamic_generic.o $(OBJDIR)/filter_registry.o \
										$(OBJDIR)/aqf_adapter.o $(OBJDIR)/stacked_adapter.o $(OBJDIR)/stacked_wrapper.o \
										$(OBJDIR)/bloom_adapter.o $(OBJDIR)/bloom_wrapper.o \
										$(OBJDIR)/lbf_adapter.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
										$(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/exp_utility.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o
test_shifting_generic:				$(OBJDIR)/test_shifting_generic.o $(OBJDIR)/filter_registry.o \
										$(OBJDIR)/aqf_adapter.o $(OBJDIR)/stacked_adapter.o $(OBJDIR)/stacked_wrapper.o \
										$(OBJDIR)/bloom_adapter.o $(OBJDIR)/bloom_wrapper.o \
										$(OBJDIR)/lbf_adapter.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
										$(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/exp_utility.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_splinter_parallel:						$(OBJDIR)/test_splinter_parallel.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_parallel:						$(OBJDIR)/test_parallel.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/splinter_util.o $(OBJDIR)/test_driver.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

test_progress:								$(OBJDIR)/test_progress.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_throughput_old:							$(OBJDIR)/test_throughput_old.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_micro_throughput:							$(OBJDIR)/test_micro_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_micro_write:							$(OBJDIR)/test_micro_write.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_micro_read:							$(OBJDIR)/test_micro_read.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_lltable_throughput:							$(OBJDIR)/test_lltable_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_fill_varied_throughput:						$(OBJDIR)/test_fill_varied_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_near_full:								$(OBJDIR)/test_near_full.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_deletions:								$(OBJDIR)/test_deletions.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_merge:								$(OBJDIR)/test_merge.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_bulk:								$(OBJDIR)/test_bulk.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_whitelist:								$(OBJDIR)/test_whitelist.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_hash_accesses:							$(OBJDIR)/test_hash_accesses.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_resize:							$(OBJDIR)/test_resize.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_threadsafe:							$(OBJDIR)/test_threadsafe.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_pc:								$(OBJDIR)/test_partitioned_counter.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

bm:										$(OBJDIR)/bm.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_ext_throughput:							$(OBJDIR)/test_ext_throughput.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_ext_inc_throughput:							$(OBJDIR)/test_ext_inc_throughput.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_zipf_throughput:							$(OBJDIR)/test_zipf_throughput.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_ext_churn:							$(OBJDIR)/test_ext_churn.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

taf:									$(OBJDIR)/taf.o $(OBJDIR)/hashutil.o

test_splinter_ops:							$(OBJDIR)/test_splinter_ops.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_inserts:							$(OBJDIR)/test_splinter_inserts.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_inserts_2:							$(OBJDIR)/test_splinter_inserts_2.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_throughput:							$(OBJDIR)/test_splinter_throughput.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_zipfian_histogram:							$(OBJDIR)/test_splinter_zipfian_histogram.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_adversarial:							$(OBJDIR)/test_splinter_adversarial.o $(OBJDIR)/gqf.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o \
										$(OBJDIR)/partitioned_counter.o

test_splinter_lltable_throughput:		$(OBJDIR)/test_splinter_lltable_throughput.o $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o \
										$(OBJDIR)/zipf.o $(OBJDIR)/hashutil.o $(OBJDIR)/ll_table.o $(OBJDIR)/splinter_util.o \
										$(OBJDIR)/partitioned_counter.o $(OBJDIR)/rand_util.o

# dependencies between .o files and .h files
$(PYTARGET): $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o \
             $(OBJDIR)/ll_table.o $(OBJDIR)/partitioned_counter.o \
             $(OBJDIR)/splinter_util.o $(OBJDIR)/exp_utility.o $(OBJDIR)/rand_util.o
	$(CC) -shared -fPIC $^ -o $@ -lpthread -lssl -lcrypto -lm

$(OBJDIR)/test.o: 						$(LOC_INCLUDE)/gqf.h $(LOC_INCLUDE)/gqf_file.h \
															$(LOC_INCLUDE)/hashutil.h \
															$(LOC_INCLUDE)/partitioned_counter.h

$(OBJDIR)/test_threadsafe.o: 	$(LOC_INCLUDE)/gqf.h $(LOC_INCLUDE)/gqf_file.h \
															$(LOC_INCLUDE)/hashutil.h \
															$(LOC_INCLUDE)/partitioned_counter.h

$(OBJDIR)/bm.o:								$(LOC_INCLUDE)/gqf_wrapper.h \
															$(LOC_INCLUDE)/partitioned_counter.h


# dependencies between .o files and .cc (or .c) files

$(OBJDIR)/gqf.o:						$(LOC_SRC)/gqf.c $(LOC_INCLUDE)/gqf.h
$(OBJDIR)/gqf_file.o:					$(LOC_SRC)/gqf_file.c $(LOC_INCLUDE)/gqf_file.h
$(OBJDIR)/hashutil.o:					$(LOC_SRC)/hashutil.c $(LOC_INCLUDE)/hashutil.h
$(OBJDIR)/partitioned_counter.o:		$(LOC_INCLUDE)/partitioned_counter.h
$(OBJDIR)/ll_table.o:					$(LOC_SRC)/ll_table.c $(LOC_INCLUDE)/ll_table.h
$(OBJDIR)/splinter_util.o:				$(LOC_SRC)/splinter_util.c $(LOC_INCLUDE)/splinter_util.h# $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o
$(OBJDIR)/test_driver.o:				$(LOC_SRC)/test_driver.c $(LOC_INCLUDE)/test_driver.h# $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/splinter_util.o
$(OBJDIR)/write_queries.o:				$(LOC_TEST)/write_queries.c $(LOC_SRC)/rand_util.c $(LOC_INCLUDE)/rand_util.h
$(OBJDIR)/exp_utility.o:				$(LOC_SRC)/exp_utility.c $(LOC_INCLUDE)/exp_utility.h
$(OBJDIR)/filter_registry.o:			$(LOC_SRC)/filter_registry.c $(LOC_INCLUDE)/filter_iface.h
$(OBJDIR)/aqf_adapter.o:				$(LOC_SRC)/aqf_adapter.c $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h
$(OBJDIR)/stacked_adapter.o:			$(LOC_SRC)/stacked_adapter.c $(LOC_INCLUDE)/filter_iface.h $(WRAPPER)/stacked_wrapper.h
$(OBJDIR)/test_distribution_generic.o:	$(LOC_TEST)/test_distribution_generic.c $(LOC_INCLUDE)/harness_output.h $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h
$(OBJDIR)/test_one_pass_generic.o:	$(LOC_TEST)/test_one_pass_generic.c $(LOC_INCLUDE)/harness_output.h $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h
$(OBJDIR)/test_advers_dist_generic.o:	$(LOC_TEST)/test_advers_dist_generic.c $(LOC_INCLUDE)/harness_output.h $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h
$(OBJDIR)/test_dynamic_generic.o:	$(LOC_TEST)/test_dynamic_generic.c $(LOC_INCLUDE)/harness_output.h $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h
# the shifting workload is the dynamic harness built with SHIFTING_WORKLOAD
$(OBJDIR)/test_shifting_generic.o:	$(LOC_TEST)/test_dynamic_generic.c $(LOC_INCLUDE)/harness_output.h $(LOC_INCLUDE)/filter_iface.h $(LOC_INCLUDE)/exp_utility.h | $(OBJDIR)
	$(CC) $(CXXFLAGS) -DSHIFTING_WORKLOAD $(INCLUDE) $< -c -o $@

$(OBJDIR)/bloom_adapter.o:				$(LOC_SRC)/bloom_adapter.c $(LOC_INCLUDE)/filter_iface.h $(BLOOM_WRAPPER)/bloom_wrapper.h
# compiled with only the fastfilter include path: its bloom.h includes "hashutil.h", which must not
# resolve to include/hashutil.h
$(OBJDIR)/bloom_wrapper.o:				$(BLOOM_WRAPPER)/bloom_wrapper.cpp $(BLOOM_WRAPPER)/bloom_wrapper.h $(FASTFILTER)/bloom/bloom.h | $(OBJDIR)
											$(CXX) -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I$(BLOOM_WRAPPER) -I$(FASTFILTER) -c $< -o $@

$(OBJDIR)/lbf_adapter.o:				$(LOC_SRC)/lbf_adapter.c $(LOC_INCLUDE)/filter_iface.h $(CASCADED_WRAPPER)/lbf_wrapper.h
$(OBJDIR)/lib%_wrapper.so:				$(CASCADED_WRAPPER)/%_wrapper.cpp $(CASCADED_WRAPPER)/lbf_wrapper.h $(CASCADED_WRAPPER)/lbf_common.h | $(OBJDIR)
											$(CXX) $(LBF_CXXFLAGS) -shared $< -o $@ -Wl,-soname,lib$*_wrapper.so \
												$(XGBOOST_LIB)/libxgboost.so -Wl,-Bsymbolic -Wl,-rpath,$(XGBOOST_LIB)

$(OBJDIR)/stacked_wrapper.o:			$(WRAPPER)/stacked_wrapper.cpp $(WRAPPER)/stacked_wrapper.h
											$(CXX) $(CXXFLAGS) -c $< -o $@

# generic build rules
#

$(CTARGETS):
	$(LD) $^ -o $@ $(LDFLAGS)

$(SPLTARGETS):
	$(LD) $^ -o $@ $(LDFLAGS)

$(CXXTARGETS):
	$(CXX) $^ -o $@ $(CXXFLAGS)

# objects that time operations or track false positive sources are compiled again for the fpsource variants
FPSOURCE_OBJS=$(FPSOURCE_OBJDIR)/filter_registry.o $(FPSOURCE_OBJDIR)/aqf_adapter.o $(FPSOURCE_OBJDIR)/stacked_adapter.o \
			  $(FPSOURCE_OBJDIR)/bloom_adapter.o $(FPSOURCE_OBJDIR)/lbf_adapter.o $(FPSOURCE_OBJDIR)/exp_utility.o
SHARED_HARNESS_OBJS=$(OBJDIR)/stacked_wrapper.o $(OBJDIR)/bloom_wrapper.o $(OBJDIR)/libplbf_wrapper.so $(OBJDIR)/libclbf_wrapper.so \
			  $(OBJDIR)/gqf.o $(OBJDIR)/gqf_file.o $(OBJDIR)/hashutil.o $(OBJDIR)/partitioned_counter.o $(OBJDIR)/ll_table.o $(OBJDIR)/rand_util.o

$(FPSOURCE_TARGETS): %_fpsource: $(FPSOURCE_OBJDIR)/%.o $(FPSOURCE_OBJS) $(SHARED_HARNESS_OBJS)
	$(LD) $^ -o $@ $(LDFLAGS)

$(FPSOURCE_OBJDIR)/%.o: $(LOC_SRC)/%.c $(wildcard $(LOC_INCLUDE)/*.h) | $(FPSOURCE_OBJDIR)
	$(CC) $(CXXFLAGS) -DTRACK_FP_SOURCE -fPIC $(INCLUDE) $< -c -o $@

$(FPSOURCE_OBJDIR)/%.o: $(LOC_TEST)/%.c $(wildcard $(LOC_INCLUDE)/*.h) | $(FPSOURCE_OBJDIR)
	$(CC) $(CXXFLAGS) -DTRACK_FP_SOURCE $(INCLUDE) $< -c -o $@

$(FPSOURCE_OBJDIR)/test_shifting_generic.o: $(LOC_TEST)/test_dynamic_generic.c $(wildcard $(LOC_INCLUDE)/*.h) | $(FPSOURCE_OBJDIR)
	$(CC) $(CXXFLAGS) -DTRACK_FP_SOURCE -DSHIFTING_WORKLOAD $(INCLUDE) $< -c -o $@

$(FPSOURCE_OBJDIR):
	@mkdir -p $(FPSOURCE_OBJDIR)

$(OBJDIR)/%.o: $(LOC_SRC)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR)/%.o: $(LOC_SRC)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) -fPIC $(INCLUDE) $< -c -o $@

$(OBJDIR)/%.o: $(LOC_TEST)/%.c | $(OBJDIR)
	$(CC) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(CTARGETS) $(FPSOURCE_TARGETS) $(CXXTARGETS) $(SPLTARGETS) core
