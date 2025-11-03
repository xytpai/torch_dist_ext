#pragma once

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "collectives.h"
#include "compare.h"
#include "device_common.h"

template <typename DataType>
void fill_zeros(DataType *ptr, unsigned int len) {
    const DataType val = 0;
    for (unsigned int i = 0; i < len; i++) ptr[i] = val;
}

template <typename DataType>
void fill_ones(DataType *ptr, unsigned int len) {
    const DataType val = 1;
    for (unsigned int i = 0; i < len; i++) ptr[i] = val;
}

template <typename DataType>
void fill_values(DataType *ptr, unsigned int len, const DataType val) {
    for (unsigned int i = 0; i < len; i++) ptr[i] = val;
}

template <typename DataType>
void fill_rand(DataType *ptr, unsigned int len, const DataType lower, const DataType upper) {
    int diff = upper - lower;
    for (unsigned int i = 0; i < len; i++) ptr[i] = lower + (rand() % diff);
}

template <>
void fill_rand<int>(int *ptr, unsigned int len, const int lower, const int upper) {
    int diff = upper - lower;
    for (unsigned int i = 0; i < len; i++) ptr[i] = lower + (rand() % diff);
}

template <>
void fill_rand<float>(float *ptr, unsigned int len, const float lower, const float upper) {
    float diff = upper - lower;
    for (unsigned int i = 0; i < len; i++) ptr[i] = lower + diff * (rand() / (float)INT_MAX);
}

template <>
void fill_rand<double>(double *ptr, unsigned int len, const double lower, const double upper) {
    float diff = upper - lower;
    for (unsigned int i = 0; i < len; i++) ptr[i] = lower + diff * (rand() / (double)INT_MAX);
}

int randint_scalar(const int lower, const int upper) {
    int diff = upper - lower;
    return lower + (rand() % diff);
}

class HostBarrier {
public:
    explicit HostBarrier(int count) :
        count(count), arrived(0), generation(0) {
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        if (++arrived == count) {
            arrived = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return gen != generation; });
        }
    }
    int gen() const {
        return generation;
    }
    std::mutex mtx;

private:
    std::condition_variable cv;
    int count;
    int arrived;
    int generation;
};
