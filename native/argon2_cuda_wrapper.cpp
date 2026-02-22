#include "argon2_cuda_wrapper.h"

#include "argon2-cuda/cudaexception.h"
#include "argon2-cuda/device.h"
#include "argon2-cuda/globalcontext.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/programcontext.h"
#include "argon2-gpu-common/argon2-common.h"
#include "argon2-gpu-common/argon2params.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using argon2::Argon2Params;
using argon2::ARGON2_D;
using argon2::ARGON2_VERSION_13;
using argon2::cuda::CudaException;
using argon2::cuda::Device;
using argon2::cuda::GlobalContext;
using argon2::cuda::ProcessingUnit;
using argon2::cuda::ProgramContext;

struct argon2_gpu_job {
    GlobalContext global;
    Device device;
    std::vector<uint8_t> salt;
    Argon2Params params;
    ProgramContext program;
    ProcessingUnit processing;
    size_t batch_size;

    argon2_gpu_job(
        int device_index,
        const uint8_t *salt_data,
        size_t salt_len,
        uint32_t m_cost,
        uint32_t t_cost,
        uint32_t lanes,
        size_t batch)
        : global(),
          device(device_index),
          salt(salt_data, salt_data + salt_len),
          params(
              32,
              salt.data(),
              salt.size(),
              nullptr,
              0,
              nullptr,
              0,
              t_cost,
              m_cost,
              lanes),
          program(&global, std::vector<Device>{device}, ARGON2_D, ARGON2_VERSION_13),
          processing(&program, &params, &device, batch, false, false),
          batch_size(batch) {}
};

int argon2_gpu_device_count(void) {
    try {
        GlobalContext global;
        return static_cast<int>(global.getAllDevices().size());
    } catch (...) {
        return -1;
    }
}

int argon2_gpu_device_name(int index, char *out, size_t out_len) {
    if (!out || out_len == 0) {
        return -1;
    }
    try {
        GlobalContext global;
        const auto &devices = global.getAllDevices();
        if (index < 0 || static_cast<size_t>(index) >= devices.size()) {
            return -2;
        }
        std::string name = devices[static_cast<size_t>(index)].getName();
        size_t to_copy = std::min(out_len - 1, name.size());
        std::memcpy(out, name.data(), to_copy);
        out[to_copy] = '\0';
        return static_cast<int>(to_copy);
    } catch (...) {
        return -3;
    }
}

argon2_gpu_job *argon2_gpu_job_create(
    int device_index,
    const uint8_t *salt,
    size_t salt_len,
    uint32_t m_cost,
    uint32_t t_cost,
    uint32_t lanes,
    size_t batch_size) {
    if (!salt || salt_len == 0 || batch_size == 0) {
        return nullptr;
    }
    try {
        GlobalContext global;
        const auto &devices = global.getAllDevices();
        if (device_index < 0 || static_cast<size_t>(device_index) >= devices.size()) {
            return nullptr;
        }
        return new argon2_gpu_job(
            device_index, salt, salt_len, m_cost, t_cost, lanes, batch_size);
    } catch (const CudaException &) {
        return nullptr;
    } catch (const std::exception &) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void argon2_gpu_job_destroy(argon2_gpu_job *job) {
    delete job;
}

size_t argon2_gpu_job_batch_size(const argon2_gpu_job *job) {
    if (!job) {
        return 0;
    }
    return job->batch_size;
}

int argon2_gpu_hash_batch(
    argon2_gpu_job *job,
    const uint8_t *pw_base,
    size_t pw_stride,
    const uint32_t *pw_lens,
    size_t batch,
    uint8_t *out_hashes) {
    if (!job || !pw_base || !pw_lens || !out_hashes) {
        return -1;
    }
    if (batch == 0 || batch > job->batch_size) {
        return -2;
    }

    try {
        for (size_t i = 0; i < batch; i++) {
            const uint8_t *pw = pw_base + (i * pw_stride);
            job->processing.setPassword(i, pw, pw_lens[i]);
        }

        job->processing.beginProcessing();
        job->processing.endProcessing();

        for (size_t i = 0; i < batch; i++) {
            job->processing.getHash(i, out_hashes + (i * 32));
        }
    } catch (const CudaException &) {
        return -3;
    } catch (const std::exception &) {
        return -4;
    } catch (...) {
        return -5;
    }

    return 0;
}
