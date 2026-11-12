/**
 * @file csim.c
 * @brief Simulate a set-associative, write-back cache from a memory trace.
 *
 * Usage: ./csim -s <set bits> -E <lines per set> -b <block bits>
 *               -t <trace file>
 * Use -h to display the command-line help. The trace contains load (L)
 * and store (S) operations. After processing a valid trace, the program
 * reports hits, misses, evictions, and dirty-byte statistics.
 *
 * Cache lines are stored in one heap-allocated array. Each group of E
 * consecutive elements represents one set. A valid line is identified
 * by its tag; last_used records access order for LRU replacement.
 * Dirty-byte counts describe whole cache blocks, even when a trace
 * operation accesses fewer bytes. The lab guarantees that each access
 * affects only one block, so simulation uses its address, not its size.
 *
 * The trace parser rejects malformed records before simulating them.
 * Statistics are printed only if the entire trace was processed
 * successfully.
 */

#include "cachelab.h"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/** Maximum number of bytes read for one trace line, including '\0'. */
#define LINELEN 256

/**
 * One cache line. A line with valid == false is available for a new block.
 * Dirty lines must be counted as written back when they are evicted.
 */
typedef struct {
    bool valid;
    bool dirty;
    uint64_t tag;
    uint64_t last_used;
} cache_line;

/**
 * State shared by trace processing and cache simulation.
 *
 * lines points to a flat array of 2^s sets, each containing E lines.
 * clock increases after every access and supplies timestamps for LRU;
 * stats accumulates the results reported by printSummary.
 */
typedef struct {
    cache_line *lines;
    int s;
    int E;
    int b;
    csim_stats_t stats;
    uint64_t clock;
} cache_sim;

/**
 * Simulate one load or store to the block containing addr.
 *
 * A hit refreshes the line's LRU timestamp. On a miss, an invalid line
 * is used before any valid line is evicted. Stores make a whole block
 * dirty; dirty bytes are added only when a clean line first becomes
 * dirty, and removed when a dirty line is evicted.
 */
static void simulate_access(cache_sim *sim, char op, uint64_t addr) {

    uint64_t set_index = (addr >> sim->b) & (((uint64_t)1 << sim->s) - 1);
    uint64_t tag = addr >> (sim->s + sim->b);
    unsigned long block_size = (unsigned long)((uint64_t)1 << sim->b);

    /* Sets occupy consecutive groups of E elements in the flat array. */
    cache_line *set = &sim->lines[(size_t)set_index * (size_t)sim->E];

    for (int i = 0; i < sim->E; i++) {
        cache_line *line = &set[i];

        if (line->valid && line->tag == tag) {
            sim->stats.hits++;
            line->last_used = ++sim->clock;
            if (op == 'S' && !line->dirty) {
                line->dirty = true;
                sim->stats.dirty_bytes += block_size;
            }
            return;
        }
    }

    sim->stats.misses++;

    cache_line *target = NULL;

    for (int i = 0; i < sim->E; i++) {
        if (!set[i].valid) {
            target = &set[i];
            break;
        }
    }

    if (target == NULL) {
        target = &set[0];

        for (int i = 1; i < sim->E; i++) {
            if (set[i].last_used < target->last_used) {
                target = &set[i];
            }
        }

        sim->stats.evictions++;

        if (target->dirty) {
            sim->stats.dirty_evictions += block_size;
            sim->stats.dirty_bytes -= block_size;
        }
    }

    target->valid = true;
    target->tag = tag;
    target->last_used = ++sim->clock;
    target->dirty = (op == 'S');

    if (target->dirty) {
        sim->stats.dirty_bytes += block_size;
    }
}

/** Print the accepted command-line options and required arguments. */
void help(char *c) {
    printf("Usage: %s [-v] -s <s> -b <b> -E <E> -t <trace>\n"
           "       %s -h\n\n"
           "-h          Print this help message and exit\n"
           "-s <s>      Number of set index bits (there are 2**s sets)\n"
           "-b <b>      Number of block bits (there are 2**b blocks)\n"
           "-E <E>      Number of lines per set (associativity)\n"
           "-t <trace>  File name of the memory trace to process\n\n"
           "The -s, -b, -E, and -t options must be supplied for all "
           "simulations.\n",
           c, c);
}

/**
 * Read, validate, and simulate the operations in a trace file.
 *
 * Each record supplies an L or S operation, a hexadecimal address, and
 * a decimal size. The size is checked as input but is not needed by the
 * simulator because each lab trace access affects only one cache block.
 * Return 0 on success or 1 for a file or parsing error.
 */
static int process_trace_file(const char *trace, cache_sim *sim) {

    FILE *tfp = fopen(trace, "r");
    if (tfp == NULL) {
        fprintf(stderr, "Error opening '%s': %s\n", trace, strerror(errno));
        return 1;
    }
    char linebuf[LINELEN];
    int parse_error = 0;
    unsigned long line_number = 0;

    while (fgets(linebuf, LINELEN, tfp)) {
        line_number++;

        /* A full buffer without a newline may contain only part of a line. */
        size_t length = strlen(linebuf);
        if (length == LINELEN - 1 && linebuf[length - 1] != '\n') {
            int next = fgetc(tfp);
            if (next != EOF) {
                fprintf(stderr, "%s:%lu: Line too long\n", trace, line_number);
                parse_error = 1;
                break;
            }
        }

        char op = linebuf[0];
        if ((op != 'L' && op != 'S') || linebuf[1] != ' ') {
            fprintf(stderr, "%s:%lu: Invalid operation\n", trace, line_number);
            parse_error = 1;
            break;
        }

        char *start = linebuf + 2;
        char *end;

        if (!isxdigit((unsigned char)*start)) {
            fprintf(stderr, "%s:%lu: Invalid address\n", trace, line_number);
            parse_error = 1;
            break;
        }

        errno = 0;
        unsigned long long parsed_addr = strtoull(start, &end, 16);
        if (errno == ERANGE || parsed_addr > UINT64_MAX || *end != ',') {
            fprintf(stderr, "%s:%lu: Invalid address\n", trace, line_number);
            parse_error = 1;
            break;
        }

        /* strtoull leaves end at the comma; Size begins after it. */
        start = end + 1;

        errno = 0;
        unsigned long size = strtoul(start, &end, 10);

        while (*end == ' ' || *end == '\t' || *end == '\r')
            end++;

        if (errno == ERANGE || size == 0 || (*end != '\n' && *end != '\0')) {
            fprintf(stderr, "%s:%lu: Invalid size or trailing text\n", trace,
                    line_number);
            parse_error = 1;
            break;
        }

        simulate_access(sim, op, parsed_addr);
    }

    if (ferror(tfp)) {
        fprintf(stderr, "Error reading '%s'\n", trace);
        parse_error = 1;
    }

    if (fclose(tfp) != 0) {
        fprintf(stderr, "Error closing '%s'\n", trace);
        parse_error = 1;
    }

    return parse_error;
}

/**
 * Parse cache parameters, allocate and initialize its lines, process
 * the requested trace, and print the resulting statistics.
 *
 * Return a nonzero status for invalid input, allocation failure, or
 * trace-processing failure. The cache allocation is freed on either
 * outcome after trace processing.
 */
int main(int argc, char *argv[]) {

    int s = -1;
    int E = -1;
    int b = -1;

    const char *trace = NULL;
    int opt;

    while ((opt = getopt(argc, argv, "s:E:b:t:h")) != -1) {
        switch (opt) {
        case 's':
            s = atoi(optarg);
            break;
        case 'E':
            E = atoi(optarg);
            break;
        case 'b':
            b = atoi(optarg);
            break;
        case 't':
            trace = optarg;
            break;
        case 'h':
            help(argv[0]);
            return 0;
        default:
            fprintf(stderr, "Extra arguments passed.\n");
            help(argv[0]);
            return 1;
        }
    }
    if (optind != argc) {
        fprintf(stderr, "Extra arguments passed.\n");
        help(argv[0]);
        return 1;
    }

    if (s < 0 || E <= 0 || b < 0 || trace == NULL || optind != argc) {
        fprintf(stderr, "Mandatory arguments missing or zero.\n");
        help(argv[0]);
        return 1;
    }

    if (s >= 64 || b >= 64 || s > 63 - b) {
        fprintf(stderr,
                "Invalid cache configuration: s + b must be less than 64.\n");
        help(argv[0]);
        return 1;
    }

    size_t set_count = (size_t)1 << s;
    size_t line_count = set_count * (size_t)E;

    /* Zero initialization makes every line invalid and every count zero. */
    cache_line *cache = calloc(line_count, sizeof(*cache));
    if (cache == NULL) {
        fprintf(stderr, "Cannot allocate cache\n");
        return 1;
    }

    cache_sim sim = {.lines = cache, .s = s, .E = E, .b = b};

    int result = process_trace_file(trace, &sim);

    if (result == 0) {
        printSummary(&sim.stats);
    }

    free(cache);

    return result;
}
