#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <gmp.h>
#include <openssl/evp.h>
#include <openssl/provider.h>
#include <openssl/err.h>
#include <secp256k1.h>
#include <sched.h>
#include <sys/time.h>
#include <immintrin.h>

#define NUM_CORES 60
#define DP_BITS 16
#define MAX_POINTS 500000
#define SPEED_UPDATE_INTERVAL 5000000
#define BATCH_SIZE 8  // AVX2 обрабатывает 8 32-битных значений одновременно

typedef struct {
    mpz_t x;
    unsigned char hash160[20];
} Point;

typedef struct {
    int core_id;
    mpz_t range_start;
    mpz_t range_end;
    unsigned char target_hash160[20];
    Point *tame;
    Point *wild;
    size_t tame_count;
    size_t wild_count;
    int found;
    int fail;
    mpz_t result_key;
    FILE *logfile;
    
    uint64_t total_operations;
    uint64_t last_operations;
    struct timeval last_time;
    double current_speed;
} ThreadData;

pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
uint64_t global_total_operations = 0;
double global_total_speed = 0.0;
struct timeval start_time;

secp256k1_context *ctx;
mpz_t order, lambda;
OSSL_PROVIDER *legacy_provider = NULL;
OSSL_PROVIDER *default_provider = NULL;
const char *range_start = "400000000000000000";
const char *range_end = "7fffffffffffffffff";
unsigned char target_hash160[20] = {
    0xf6, 0xf5, 0x43, 0x1d, 0x25, 0xbb, 0xf7, 0xb1, 0x2e, 0x8a,
    0xdd, 0x9a, 0xf5, 0xe3, 0x47, 0x5c, 0x44, 0xa0, 0xa5, 0xb8
};

double time_diff(struct timeval *start, struct timeval *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_usec - start->tv_usec) / 1000000.0;
}

void format_number(uint64_t num, char *buffer) {
    if (num >= 1000000000000ULL) {
        sprintf(buffer, "%.2fT", num / 1000000000000.0);
    } else if (num >= 1000000000ULL) {
        sprintf(buffer, "%.2fG", num / 1000000000.0);
    } else if (num >= 1000000ULL) {
        sprintf(buffer, "%.2fM", num / 1000000.0);
    } else if (num >= 1000ULL) {
        sprintf(buffer, "%.2fK", num / 1000.0);
    } else {
        sprintf(buffer, "%llu", (unsigned long long)num);
    }
}

void format_speed(double speed, char *buffer) {
    if (speed >= 1000000000.0) {
        sprintf(buffer, "%.2f GH/s", speed / 1000000000.0);
    } else if (speed >= 1000000.0) {
        sprintf(buffer, "%.2f MH/s", speed / 1000000.0);
    } else if (speed >= 1000.0) {
        sprintf(buffer, "%.2f KH/s", speed / 1000.0);
    } else {
        sprintf(buffer, "%.2f H/s", speed);
    }
}

// Оптимизированная функция проверки distinguished points с AVX2
int is_distinguished_batch_avx2(unsigned char hash160_batch[][20], int batch_size, int *results) {
    __m256i dp_mask = _mm256_set1_epi32((1 << DP_BITS) - 1);
    __m256i zeros = _mm256_setzero_si256();
    
    for (int i = 0; i < batch_size; i += 8) {
        __m256i vals = _mm256_setzero_si256();
        
        // Загружаем значения из hash160
        for (int j = 0; j < 8 && (i + j) < batch_size; j++) {
            uint32_t val = 0;
            for (int k = 19 - (DP_BITS / 8); k < 20; k++) {
                val = (val << 8) | hash160_batch[i + j][k];
            }
            if (DP_BITS % 8) {
                val >>= (8 - (DP_BITS % 8));
            }
            vals = _mm256_insert_epi32(vals, val, j);
        }
        
        // Применяем маску и сравниваем с нулем
        __m256i masked = _mm256_and_si256(vals, dp_mask);
        __m256i cmp = _mm256_cmpeq_epi32(masked, zeros);
        
        // Извлекаем результаты
        int mask = _mm256_movemask_epi8(cmp);
        for (int j = 0; j < 8 && (i + j) < batch_size; j++) {
            results[i + j] = (mask & (0xF << (j * 4))) != 0;
        }
    }
    
    return 1;
}

// Оптимизированная функция сравнения хешей с AVX2
int compare_hashes_avx2(unsigned char *hash1, unsigned char *hash2) {
    __m256i h1 = _mm256_loadu_si256((__m256i*)hash1);
    __m256i h2 = _mm256_loadu_si256((__m256i*)hash2);
    __m256i cmp = _mm256_cmpeq_epi8(h1, h2);
    
    // Проверяем только первые 20 байт (остальные 12 игнорируем)
    uint32_t mask = _mm256_movemask_epi8(cmp) & 0xFFFFF;
    return mask == 0xFFFFF;
}

// Batch обработка нескольких хешей одновременно
int batch_hash160_avx2(mpz_t *privs, unsigned char hash160_batch[][20], int batch_size, FILE *logfile) {
    // Обрабатываем все приватные ключи в пакете
    for (int i = 0; i < batch_size; i++) {
        unsigned char priv_bytes[32] = {0};
        size_t priv_len = 32;
        mpz_export(priv_bytes, &priv_len, 1, 1, 0, 0, privs[i]);
        if (priv_len < 32) {
            memmove(priv_bytes + (32 - priv_len), priv_bytes, priv_len);
            memset(priv_bytes, 0, 32 - priv_len);
        }

        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, priv_bytes)) {
            return 0;
        }

        unsigned char compressed[33] = {0};
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(ctx, compressed, &len, &pubkey, SECP256K1_EC_COMPRESSED);

        unsigned char sha256[32];
        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) return 0;

        if (!EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) ||
            !EVP_DigestUpdate(mdctx, compressed, 33) ||
            !EVP_DigestFinal_ex(mdctx, sha256, NULL)) {
            EVP_MD_CTX_free(mdctx);
            return 0;
        }

        if (!EVP_DigestInit_ex(mdctx, EVP_ripemd160(), NULL) ||
            !EVP_DigestUpdate(mdctx, sha256, 32) ||
            !EVP_DigestFinal_ex(mdctx, hash160_batch[i], NULL)) {
            EVP_MD_CTX_free(mdctx);
            return 0;
        }
        EVP_MD_CTX_free(mdctx);
    }
    return 1;
}

int compute_hash160(mpz_t priv, unsigned char *hash160, FILE *logfile) {
    if (!priv || !hash160 || !logfile) {
        fprintf(logfile, "Error: Null pointer in compute_hash160\n");
        fflush(logfile);
        return 0;
    }

    unsigned char priv_bytes[32] = {0};
    size_t priv_len = 32;
    mpz_export(priv_bytes, &priv_len, 1, 1, 0, 0, priv);
    if (priv_len < 32) {
        memmove(priv_bytes + (32 - priv_len), priv_bytes, priv_len);
        memset(priv_bytes, 0, 32 - priv_len);
    }

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, priv_bytes)) {
        fprintf(logfile, "Error: Failed to create public key\n");
        fflush(logfile);
        return 0;
    }

    unsigned char compressed[33] = {0};
    size_t len = 33;
    secp256k1_ec_pubkey_serialize(ctx, compressed, &len, &pubkey, SECP256K1_EC_COMPRESSED);

    unsigned char sha256[32];
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    if (!mdctx) {
        fprintf(logfile, "Error: EVP_MD_CTX_new failed\n");
        fflush(logfile);
        return 0;
    }

    if (!EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL) ||
        !EVP_DigestUpdate(mdctx, compressed, 33) ||
        !EVP_DigestFinal_ex(mdctx, sha256, NULL)) {
        fprintf(logfile, "Error: SHA256 digest failed: %s\n", ERR_error_string(ERR_get_error(), NULL));
        EVP_MD_CTX_free(mdctx);
        fflush(logfile);
        return 0;
    }

    if (!EVP_DigestInit_ex(mdctx, EVP_ripemd160(), NULL) ||
        !EVP_DigestUpdate(mdctx, sha256, 32) ||
        !EVP_DigestFinal_ex(mdctx, hash160, NULL)) {
        fprintf(logfile, "Error: RIPEMD160 digest failed: %s\n", ERR_error_string(ERR_get_error(), NULL));
        EVP_MD_CTX_free(mdctx);
        fflush(logfile);
        return 0;
    }
    EVP_MD_CTX_free(mdctx);
    return 1;
}

int is_distinguished(unsigned char *hash160) {
    uint32_t val = 0;
    for (int i = 19 - (DP_BITS / 8); i < 20; i++) {
        val = (val << 8) | hash160[i];
    }
    if (DP_BITS % 8) {
        val >>= (8 - (DP_BITS % 8));
    }
    return (val & ((1 << DP_BITS) - 1)) == 0;
}

void jump_batch_avx2(mpz_t *x_batch, unsigned char hash160_batch[][20], uint64_t *jump_batch, int batch_size, FILE *logfile) {
    mpz_t tmp;
    mpz_init(tmp);
    
    for (int i = 0; i < batch_size; i++) {
        mpz_add_ui(tmp, x_batch[i], jump_batch[i]);
        mpz_mod(tmp, tmp, order);
        mpz_set(x_batch[i], tmp);
    }
    
    batch_hash160_avx2(x_batch, hash160_batch, batch_size, logfile);
    mpz_clear(tmp);
}

void update_speed(ThreadData *data, uint64_t operations) {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    
    double elapsed = time_diff(&data->last_time, &current_time);
    if (elapsed > 0) {
        uint64_t ops_diff = operations - data->last_operations;
        data->current_speed = ops_diff / elapsed;
        
        pthread_mutex_lock(&stats_mutex);
        global_total_operations += ops_diff;
        global_total_speed += data->current_speed;
        pthread_mutex_unlock(&stats_mutex);
        
        data->last_operations = operations;
        data->last_time = current_time;
    }
}

void *kangaroo_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    mpz_t x_t_batch[BATCH_SIZE], x_w_batch[BATCH_SIZE], jump_size;
    unsigned char hash160_t_batch[BATCH_SIZE][20], hash160_w_batch[BATCH_SIZE][20];
    uint64_t jumps[64];
    uint64_t jump_batch_t[BATCH_SIZE], jump_batch_w[BATCH_SIZE];
    int dp_results_t[BATCH_SIZE], dp_results_w[BATCH_SIZE];
    gmp_randstate_t rng;

    for (int i = 0; i < BATCH_SIZE; i++) {
        mpz_init(x_t_batch[i]);
        mpz_init(x_w_batch[i]);
    }
    mpz_init(jump_size);
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, time(NULL) ^ data->core_id);

    data->total_operations = 0;
    data->last_operations = 0;
    data->current_speed = 0.0;
    gettimeofday(&data->last_time, NULL);

    for (int i = 0; i < 64; i++) jumps[i] = 1ULL << i;

    // Инициализация batch tame
    for (int i = 0; i < BATCH_SIZE; i++) {
        mpz_set(x_t_batch[i], data->range_end);
        mpz_add_ui(x_t_batch[i], x_t_batch[i], i); // Небольшое смещение для разнообразия
    }
    
    if (!batch_hash160_avx2(x_t_batch, hash160_t_batch, BATCH_SIZE, data->logfile)) {
        fprintf(data->logfile, "Core %d: Tame batch init failed\n", data->core_id);
        data->fail = 1;
        return NULL;
    }

    // Инициализация batch wild
    mpz_sub(jump_size, data->range_end, data->range_start);
    for (int i = 0; i < BATCH_SIZE; i++) {
        mpz_urandomm(x_w_batch[i], rng, jump_size);
        mpz_add(x_w_batch[i], x_w_batch[i], data->range_start);
    }
    
    if (!batch_hash160_avx2(x_w_batch, hash160_w_batch, BATCH_SIZE, data->logfile)) {
        fprintf(data->logfile, "Core %d: Wild batch init failed\n", data->core_id);
        data->fail = 1;
        return NULL;
    }

    fprintf(data->logfile, "Core %d: Initialized with batch size %d\n", data->core_id, BATCH_SIZE);
    fflush(data->logfile);

    while (!data->found && !data->fail) {
        // Проверка distinguished points для tame batch
        is_distinguished_batch_avx2(hash160_t_batch, BATCH_SIZE, dp_results_t);
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (dp_results_t[i]) {
                if (data->tame_count < MAX_POINTS) {
                    mpz_set(data->tame[data->tame_count].x, x_t_batch[i]);
                    memcpy(data->tame[data->tame_count].hash160, hash160_t_batch[i], 20);
                    data->tame_count++;
                }
                
                if (compare_hashes_avx2(hash160_t_batch[i], data->target_hash160)) {
                    mpz_set(data->result_key, x_t_batch[i]);
                    data->found = 1;
                    fprintf(data->logfile, "Core %d: Found target key: 0x%s\n", 
                            data->core_id, mpz_get_str(NULL, 16, x_t_batch[i]));
                    break;
                }
                
                // Проверка коллизий с wild points
                for (size_t j = 0; j < data->wild_count; j++) {
                    if (compare_hashes_avx2(hash160_t_batch[i], data->wild[j].hash160)) {
                        mpz_t diff;
                        mpz_init(diff);
                        mpz_sub(diff, x_t_batch[i], data->wild[j].x);
                        mpz_mod(diff, diff, order);
                        mpz_set(data->result_key, diff);
                        data->found = 1;
                        fprintf(data->logfile, "Core %d: Collision found\n", data->core_id);
                        mpz_clear(diff);
                        break;
                    }
                }
            }
        }

        if (data->found) break;

        // Проверка distinguished points для wild batch
        is_distinguished_batch_avx2(hash160_w_batch, BATCH_SIZE, dp_results_w);
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (dp_results_w[i]) {
                if (data->wild_count < MAX_POINTS) {
                    mpz_set(data->wild[data->wild_count].x, x_w_batch[i]);
                    memcpy(data->wild[data->wild_count].hash160, hash160_w_batch[i], 20);
                    data->wild_count++;
                }
                
                if (compare_hashes_avx2(hash160_w_batch[i], data->target_hash160)) {
                    mpz_set(data->result_key, x_w_batch[i]);
                    data->found = 1;
                    fprintf(data->logfile, "Core %d: Found target key: 0x%s\n", 
                            data->core_id, mpz_get_str(NULL, 16, x_w_batch[i]));
                    break;
                }
                
                // Проверка коллизий с tame points
                for (size_t j = 0; j < data->tame_count; j++) {
                    if (compare_hashes_avx2(hash160_w_batch[i], data->tame[j].hash160)) {
                        mpz_t diff;
                        mpz_init(diff);
                        mpz_sub(diff, data->tame[j].x, x_w_batch[i]);
                        mpz_mod(diff, diff, order);
                        mpz_set(data->result_key, diff);
                        data->found = 1;
                        fprintf(data->logfile, "Core %d: Collision found\n", data->core_id);
                        mpz_clear(diff);
                        break;
                    }
                }
            }
        }

        if (data->found) break;

        // Подготовка jump размеров для batch
        for (int i = 0; i < BATCH_SIZE; i++) {
            int jump_idx_t = hash160_t_batch[i][19] % 64;
            int jump_idx_w = hash160_w_batch[i][19] % 64;
            jump_batch_t[i] = jumps[jump_idx_t];
            jump_batch_w[i] = jumps[jump_idx_w];
        }

        // Выполнение batch jumps
        jump_batch_avx2(x_t_batch, hash160_t_batch, jump_batch_t, BATCH_SIZE, data->logfile);
        jump_batch_avx2(x_w_batch, hash160_w_batch, jump_batch_w, BATCH_SIZE, data->logfile);

        data->total_operations += BATCH_SIZE * 2; // tame + wild batch

        if (data->total_operations % SPEED_UPDATE_INTERVAL == 0) {
            update_speed(data, data->total_operations);
            
            char speed_str[32];
            format_speed(data->current_speed, speed_str);
            fprintf(data->logfile, "Core %d: %llu ops, tame=%zu, wild=%zu, speed=%s\n", 
                    data->core_id, (unsigned long long)data->total_operations, 
                    data->tame_count, data->wild_count, speed_str);
            fflush(data->logfile);
        }
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        mpz_clear(x_t_batch[i]);
        mpz_clear(x_w_batch[i]);
    }
    mpz_clear(jump_size);
    gmp_randclear(rng);
    return NULL;
}

void *stats_thread(void *arg) {
    ThreadData *threads = (ThreadData *)arg;
    
    while (1) {
        sleep(10);
        
        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double total_elapsed = time_diff(&start_time, &current_time);
        
        uint64_t total_ops = 0;
        double total_speed = 0.0;
        int active_threads = 0;
        
        for (int i = 0; i < NUM_CORES; i++) {
            if (!threads[i].found && !threads[i].fail) {
                total_ops += threads[i].total_operations;
                total_speed += threads[i].current_speed;
                active_threads++;
            }
        }
        
        char ops_str[32], speed_str[32];
        format_number(total_ops, ops_str);
        format_speed(total_speed, speed_str);
        
        printf("\r[%02d:%02d:%02d] Threads: %d/%d | Operations: %s | Speed: %s | Avg: %.2f H/s/thread", 
               (int)(total_elapsed / 3600), 
               (int)((int)total_elapsed % 3600 / 60), 
               (int)total_elapsed % 60,
               active_threads, NUM_CORES, ops_str, speed_str, 
               active_threads > 0 ? total_speed / active_threads : 0.0);
        fflush(stdout);
        
        int found = 0;
        for (int i = 0; i < NUM_CORES; i++) {
            if (threads[i].found || threads[i].fail) {
                found = 1;
                break;
            }
        }
        if (found) break;
    }
    
    return NULL;
}

int main() {
    printf("Initializing AVX2-optimized Kangaroo solver with %d threads...\n", NUM_CORES);
    
    gettimeofday(&start_time, NULL);
    
    legacy_provider = OSSL_PROVIDER_load(NULL, "legacy");
    if (!legacy_provider) {
        fprintf(stderr, "Error: Failed to load OpenSSL legacy provider\n");
        return 1;
    }
    default_provider = OSSL_PROVIDER_load(NULL, "default");
    if (!default_provider) {
        fprintf(stderr, "Error: Failed to load OpenSSL default provider\n");
        OSSL_PROVIDER_unload(legacy_provider);
        return 1;
    }

    ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    mpz_inits(order, lambda, NULL);
    mpz_set_str(order, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    mpz_set_str(lambda, "5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72", 16);

    mpz_t full_start, full_end, range_size, subrange_size;
    mpz_inits(full_start, full_end, range_size, subrange_size, NULL);
    mpz_set_str(full_start, range_start, 16);
    mpz_set_str(full_end, range_end, 16);
    if (mpz_cmp(full_start, full_end) > 0) {
        fprintf(stderr, "Error: Invalid range\n");
        return 1;
    }
    mpz_sub(range_size, full_end, full_start);
    mpz_add_ui(range_size, range_size, 1);
    mpz_cdiv_q_ui(subrange_size, range_size, NUM_CORES);

    ThreadData threads[NUM_CORES];
    pthread_t tids[NUM_CORES];
    pthread_t stats_tid;

    for (int i = 0; i < NUM_CORES; i++) {
        threads[i].core_id = i;
        mpz_init(threads[i].range_start);
        mpz_init(threads[i].range_end);
        mpz_init(threads[i].result_key);
        mpz_t offset;
        mpz_init(offset);
        mpz_mul_ui(offset, subrange_size, i);
        mpz_add(threads[i].range_start, full_start, offset);
        mpz_add(threads[i].range_end, threads[i].range_start, subrange_size);
        if (i == NUM_CORES - 1 || mpz_cmp(threads[i].range_end, full_end) > 0) {
            mpz_set(threads[i].range_end, full_end);
        }
        mpz_clear(offset);
        memcpy(threads[i].target_hash160, target_hash160, 20);
        threads[i].tame = (Point *)calloc(MAX_POINTS, sizeof(Point));
        threads[i].wild = (Point *)calloc(MAX_POINTS, sizeof(Point));
        if (!threads[i].tame || !threads[i].wild) {
            fprintf(stderr, "Error: Memory allocation failed for core %d\n", i);
            return 1;
        }
        for (size_t j = 0; j < MAX_POINTS; j++) {
            mpz_init(threads[i].tame[j].x);
            mpz_init(threads[i].wild[j].x);
        }
        threads[i].tame_count = 0;
        threads[i].wild_count = 0;
        threads[i].found = 0;
        threads[i].fail = 0;
        char logname[32];
        snprintf(logname, sizeof(logname), "core_%d.log", i);
        threads[i].logfile = fopen(logname, "w");
        if (!threads[i].logfile) {
            fprintf(stderr, "Error: Failed to open log file %s\n", logname);
            return 1;
        }
    }

    printf("Starting %d worker threads with AVX2 batch processing...\n", NUM_CORES);
    
    for (int i = 0; i < NUM_CORES; i++) {
        pthread_create(&tids[i], NULL, kangaroo_thread, &threads[i]);
    }
    
    pthread_create(&stats_tid, NULL, stats_thread, threads);

    int found = 0;
    mpz_t final_key;
    mpz_init(final_key);
    
    for (int i = 0; i < NUM_CORES; i++) {
        pthread_join(tids[i], NULL);
        if (threads[i].found && !found) {
            found = 1;
            mpz_set(final_key, threads[i].result_key);
            
            printf("\n\nFOUND KEY!\n");
            
            unsigned char hash160[20];
            compute_hash160(final_key, hash160, threads[i].logfile);
            if (memcmp(hash160, target_hash160, 20) == 0) {
                printf("Found key: 0x%s\n", mpz_get_str(NULL, 16, final_key));
            } else {
                mpz_t lambda_k;
                mpz_init(lambda_k);
                mpz_mul(lambda_k, final_key, lambda);
                mpz_mod(lambda_k, lambda_k, order);
                compute_hash160(lambda_k, hash160, threads[i].logfile);
                if (memcmp(hash160, target_hash160, 20) == 0) {
                    printf("Found key (λ): 0x%s\n", mpz_get_str(NULL, 16, lambda_k));
                    mpz_set(final_key, lambda_k);
                } else {
                    mpz_mul(lambda_k, lambda_k, lambda);
                    mpz_mod(lambda_k, lambda_k, order);
                    compute_hash160(lambda_k, hash160, threads[i].logfile);
                    if (memcmp(hash160, target_hash160, 20) == 0) {
                        printf("Found key (λ²): 0x%s\n", mpz_get_str(NULL, 16, lambda_k));
                        mpz_set(final_key, lambda_k);
                    }
                }
                mpz_clear(lambda_k);
            }
        }
        if (threads[i].fail) {
            printf("\nThread %d failed!\n", i);
        }
    }
    
    pthread_cancel(stats_tid);
    pthread_join(stats_tid, NULL);

    // Очистка ресурсов
    for (int i = 0; i < NUM_CORES; i++) {
        for (size_t j = 0; j < MAX_POINTS; j++) {
            mpz_clear(threads[i].tame[j].x);
            mpz_clear(threads[i].wild[j].x);
        }
        free(threads[i].tame);
        free(threads[i].wild);
        mpz_clears(threads[i].range_start, threads[i].range_end, threads[i].result_key, NULL);
        fclose(threads[i].logfile);
    }
    
    mpz_clears(full_start, full_end, range_size, subrange_size, final_key, order, lambda, NULL);
    secp256k1_context_destroy(ctx);
    OSSL_PROVIDER_unload(legacy_provider);
    OSSL_PROVIDER_unload(default_provider);
    
    return found ? 0 : 1;
}