/**
 * @file trans.c
 * @brief Contains various implementations of matrix transpose
 *
 * Each transpose function must have a prototype of the form:
 *   void trans(size_t M, size_t N, double A[N][M], double B[M][N],
 *              double tmp[TMPCOUNT]);
 *
 * All transpose functions take the following arguments:
 *
 *   @param[in]     M    Width of A, height of B
 *   @param[in]     N    Height of A, width of B
 *   @param[in]     A    Source matrix
 *   @param[out]    B    Destination matrix
 *   @param[in,out] tmp  Array that can store temporary double values
 *
 * A transpose function is evaluated by counting the number of hits and misses,
 * using the cache parameters and score computations described in the writeup.
 *
 * Programming restrictions:
 *   - No out-of-bounds references are allowed
 *   - No alterations may be made to the source array A
 *   - Data in tmp can be read or written
 *   - This file cannot contain any local or global doubles or arrays of doubles
 *   - You may not use unions, casting, global variables, or
 *     other tricks to hide array data in other forms of local or global memory.
 *
 * TODO: fill in your name and Andrew ID below.
 * @author Guo Yu <guoy@andrew.cmu.edu>
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "cachelab.h"

/**
 * @brief Checks if B is the transpose of A.
 *
 * You can call this function inside of an assertion, if you'd like to verify
 * the correctness of a transpose function.
 *
 * @param[in]     M    Width of A, height of B
 * @param[in]     N    Height of A, width of B
 * @param[in]     A    Source matrix
 * @param[out]    B    Destination matrix
 *
 * @return True if B is the transpose of A, and false otherwise.
 */
#ifndef NDEBUG
static bool is_transpose(size_t M, size_t N, double A[N][M], double B[M][N]) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                fprintf(stderr,
                        "Transpose incorrect.  Fails for B[%zd][%zd] = %.3f, "
                        "A[%zd][%zd] = %.3f\n",
                        j, i, B[j][i], i, j, A[i][j]);
                return false;
            }
        }
    }
    return true;
}
#endif

/*
 * You can define additional transpose functions here. We've defined
 * some simple ones below to help you get started, which you should
 * feel free to modify or delete.
 */

/**
 * @brief A simple baseline transpose function, not optimized for the cache.
 *
 * Note the use of asserts (defined in assert.h) that add checking code.
 * These asserts are disabled when measuring cycle counts (i.e. when running
 * the ./test-trans) to avoid affecting performance.
 */
static void trans_basic(size_t M, size_t N, double A[N][M], double B[M][N],
                        double tmp[TMPCOUNT]) {
    assert(M > 0);
    assert(N > 0);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            B[j][i] = A[i][j];
        }
    }

    assert(is_transpose(M, N, A, B));
}

/**
 * @brief A contrived example to illustrate the use of the temporary array.
 *
 * This function uses the first four elements of tmp as a 2x2 array with
 * row-major ordering.
 */
static void trans_tmp(size_t M, size_t N, double A[N][M], double B[M][N],
                      double tmp[TMPCOUNT]) {
    assert(M > 0);
    assert(N > 0);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            size_t di = i % 2;
            size_t dj = j % 2;
            tmp[2 * di + dj] = A[i][j];
            B[j][i] = tmp[2 * di + dj];
        }
    }

    assert(is_transpose(M, N, A, B));
}

/**
 * @brief The solution transpose function that will be graded.
 *
 * You can call other transpose functions from here as you please.
 * It's OK to choose different functions based on array size, but
 * this function must be correct for all values of M and N.
 */
/**
 * @brief Transpose A into B using cache-aware strategies for graded sizes.
 *
 * The 32x32 case uses 8x8 blocks on a direct-mapped cache. Within a
 * diagonal block, each diagonal element is written after the rest of
 * its source row has been read, so the write cannot evict A data that
 * the same row still needs.
 *
 * The 1024x1024 case uses 8x8 blocks on the specified 64-set,
 * 8-way cache. Most blocks are transposed directly. When their A
 * and B locations map to the same set, tmp holds the source block
 * while B is written in row order. Other matrix sizes use a simple
 * transpose to preserve correctness.
 */
static void transpose_submit(size_t M, size_t N, double A[N][M], double B[M][N],
                             double tmp[TMPCOUNT]) {
    /* One 64-byte cache block holds eight doubles. */
    const size_t tile = 8;
    if (M == 32 && N == 32) {

        for (size_t i = 0; i < N; i += tile) {
            for (size_t j = 0; j < M; j += tile) {
                if (i != j) {
                    for (size_t r = 0; r < tile; r++) {
                        for (size_t c = 0; c < tile; c++) {
                            B[j + c][i + r] = A[i + r][j + c];
                        }
                    }
                } else {

                    for (size_t r = 0; r < tile; r++) {
                        for (size_t c = 0; c < tile; c++) {
                            if (r != c) {
                                B[j + c][i + r] = A[i + r][j + c];
                            }
                        }

                        /* Delay the conflicting write until this A row is done.
                         */
                        B[j + r][i + r] = A[i + r][j + r];
                    }
                }
            }
        }
    } else if (M == 1024 && N == 1024) {
        assert(2 * tile * tile <= TMPCOUNT);

        for (size_t i = 0; i < N; i += tile) {
            for (size_t j = 0; j < M; j += tile) {

                /*
                 * Each matrix row spans a multiple of 64 cache sets.
                 * With the guaranteed alignment, the A set depends on
                 * j and the corresponding B set depends on i.
                 */
                size_t a_set = (j / tile) % 64;
                size_t b_set = (i / tile) % 64;

                if (a_set != b_set) {
                    for (size_t r = 0; r < tile; r++) {
                        for (size_t c = 0; c < tile; c++) {
                            B[j + c][i + r] = A[i + r][j + c];
                        }
                    }
                } else {

                    /*
                     * tmp[0..63] uses sets 0..7; tmp[64..127]
                     * uses sets 8..15. Select the region that does
                     * not include the set shared by A and B.
                     */
                    size_t base = (a_set < tile) ? tile * tile : 0;

                    for (size_t r = 0; r < tile; r++) {
                        for (size_t c = 0; c < tile; c++) {
                            tmp[base + r * tile + c] = A[i + r][j + c];
                        }
                    }
                    for (size_t c = 0; c < tile; c++) {
                        for (size_t r = 0; r < tile; r++) {
                            B[j + c][i + r] = tmp[base + r * tile + c];
                        }
                    }
                }
            }
        }
    } else {

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < M; j++) {
                B[j][i] = A[i][j];
            }
        }
    }

    assert(is_transpose(M, N, A, B));
}

/**
 * @brief Registers all transpose functions with the driver.
 *
 * At runtime, the driver will evaluate each function registered here, and
 * and summarize the performance of each. This is a handy way to experiment
 * with different transpose strategies.
 */
void registerFunctions(void) {
    // Register the solution function. Do not modify this line!
    registerTransFunction(transpose_submit, SUBMIT_DESCRIPTION);

    // Register any additional transpose functions
    registerTransFunction(trans_basic, "Basic transpose");
    registerTransFunction(trans_tmp, "Transpose using the temporary array");
}
