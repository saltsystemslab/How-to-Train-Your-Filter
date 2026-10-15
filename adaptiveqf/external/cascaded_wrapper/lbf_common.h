/**
 * Helpers shared by the PLBF and CLBF wrappers (each wrapper is compiled on its own, since the
 * CascadedLBF headers for the two filters define different classes with the same names).
 */
#ifndef LBF_COMMON_H
#define LBF_COMMON_H

#include <cstdio>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

// Monotonic clock in nanoseconds.
inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// The cost of one now_ns() call, which every timed segment of a lookup includes once (its closing call) and
// which the timed lookups subtract. Measured once, as the median gap between consecutive calls.
inline uint64_t clock_overhead_ns() {
    static uint64_t overhead = [] {
        std::vector<uint64_t> gaps(10001);
        uint64_t previous = now_ns();
        for (auto &gap : gaps) {
            uint64_t now = now_ns();
            gap = now - previous;
            previous = now;
        }
        std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
        return gaps[gaps.size() / 2];
    }();
    return overhead;
}

// Elapsed time of a timed segment, without the cost of the clock call that ended it.
inline uint64_t segment_ns(uint64_t start, uint64_t end) {
    uint64_t elapsed = end - start, overhead = clock_overhead_ns();
    return elapsed > overhead ? elapsed - overhead : 0;
}

// Silences std::cout while in scope: CascadedLBF prints progress and "[INFO] Model saved" messages.
class QuietCout {
public:
    QuietCout() : old_(std::cout.rdbuf(sink_.rdbuf())) {}
    ~QuietCout() { std::cout.rdbuf(old_); }
private:
    std::ostringstream sink_;
    std::streambuf *old_;
};

// Holds an exclusive lock shared by every process using these wrappers while in scope. CascadedLBF
// writes intermediate model files to fixed paths in /tmp (e.g. /tmp/xgboost_model.json when loading or
// reducing a model), so two processes loading or configuring filters at once would corrupt each other's
// files. Lookups do not touch these files and run without the lock.
class CascadedTmpLock {
public:
    CascadedTmpLock() : fd_(open("/tmp/cascadedlbf_tmpfiles.lock", O_RDWR | O_CREAT, 0666)) {
        if (fd_ >= 0) flock(fd_, LOCK_EX);
    }
    ~CascadedTmpLock() {
        if (fd_ >= 0) {
            flock(fd_, LOCK_UN);
            close(fd_);
        }
    }
private:
    int fd_;
};

// Rows of a row-major float array as the nested vectors CascadedLBF expects.
inline std::vector<std::vector<float>> to_rows(const float *data, size_t num_rows, size_t num_cols) {
    std::vector<std::vector<float>> rows(num_rows);
    for (size_t i = 0; i < num_rows; i++) rows[i].assign(data + i * num_cols, data + (i + 1) * num_cols);
    return rows;
}

// Size in bytes of the filter's saved model, which is how CascadedLBF reports filter sizes.
template <typename Filter>
size_t saved_size_bytes(Filter &filter) {
    char path[] = "/tmp/lbf_size_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    close(fd);
    {
        QuietCout quiet;
        filter.save_model(path);
    }
    struct stat st;
    size_t size = (stat(path, &st) == 0) ? static_cast<size_t>(st.st_size) : 0;
    unlink(path);
    return size;
}

#endif
