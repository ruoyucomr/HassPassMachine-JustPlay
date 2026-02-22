#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct argon2_gpu_job argon2_gpu_job;

int argon2_gpu_device_count(void);
int argon2_gpu_device_name(int index, char *out, size_t out_len);

argon2_gpu_job *argon2_gpu_job_create(
    int device_index,
    const uint8_t *salt,
    size_t salt_len,
    uint32_t m_cost,
    uint32_t t_cost,
    uint32_t lanes,
    size_t batch_size);

void argon2_gpu_job_destroy(argon2_gpu_job *job);
size_t argon2_gpu_job_batch_size(const argon2_gpu_job *job);

int argon2_gpu_hash_batch(
    argon2_gpu_job *job,
    const uint8_t *pw_base,
    size_t pw_stride,
    const uint32_t *pw_lens,
    size_t batch,
    uint8_t *out_hashes);

#ifdef __cplusplus
} // extern "C"
#endif
