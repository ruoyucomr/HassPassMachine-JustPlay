use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;

#[repr(C)]
struct Argon2GpuJob {
    _private: [u8; 0],
}

#[link(name = "argon2_cuda_wrapper", kind = "static")]
unsafe extern "C" {
    fn argon2_gpu_device_count() -> c_int;
    fn argon2_gpu_device_name(index: c_int, out: *mut c_char, out_len: usize) -> c_int;

    fn argon2_gpu_job_create(
        device_index: c_int,
        salt: *const u8,
        salt_len: usize,
        m_cost: u32,
        t_cost: u32,
        lanes: u32,
        batch_size: usize,
    ) -> *mut Argon2GpuJob;

    fn argon2_gpu_job_destroy(job: *mut Argon2GpuJob);
    fn argon2_gpu_job_batch_size(job: *const Argon2GpuJob) -> usize;

    fn argon2_gpu_hash_batch(
        job: *mut Argon2GpuJob,
        pw_base: *const u8,
        pw_stride: usize,
        pw_lens: *const u32,
        batch: usize,
        out_hashes: *mut u8,
    ) -> c_int;
}

pub struct GpuMiner {
    job: *mut Argon2GpuJob,
    batch_size: usize,
}

unsafe impl Send for GpuMiner {}

impl GpuMiner {
    pub fn device_count() -> anyhow::Result<usize> {
        let count = unsafe { argon2_gpu_device_count() };
        if count < 0 {
            anyhow::bail!("CUDA device enumeration failed");
        }
        Ok(count as usize)
    }

    pub fn device_name(index: usize) -> anyhow::Result<String> {
        let mut buf = [0i8; 256];
        let ret = unsafe { argon2_gpu_device_name(index as c_int, buf.as_mut_ptr(), buf.len()) };
        if ret < 0 {
            anyhow::bail!("Failed to read CUDA device name for index {}", index);
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    pub fn new(
        device_index: usize,
        salt: &[u8],
        memory_cost: u32,
        time_cost: u32,
        lanes: u32,
        batch_size: usize,
    ) -> anyhow::Result<Self> {
        if salt.is_empty() {
            anyhow::bail!("salt is empty");
        }
        if batch_size == 0 {
            anyhow::bail!("batch_size must be > 0");
        }

        let job = unsafe {
            argon2_gpu_job_create(
                device_index as c_int,
                salt.as_ptr(),
                salt.len(),
                memory_cost,
                time_cost,
                lanes,
                batch_size,
            )
        };
        if job.is_null() {
            anyhow::bail!("failed to initialize CUDA miner");
        }

        let actual_batch = unsafe { argon2_gpu_job_batch_size(job) };
        if actual_batch == 0 {
            unsafe { argon2_gpu_job_destroy(job) };
            anyhow::bail!("CUDA miner returned batch size 0");
        }

        Ok(Self {
            job,
            batch_size: actual_batch,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn hash_batch(
        &mut self,
        pw_base: &[u8],
        pw_stride: usize,
        pw_lens: &[u32],
        batch: usize,
        out_hashes: &mut [u8],
    ) -> anyhow::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if batch > self.batch_size {
            anyhow::bail!("batch {} exceeds GPU batch size {}", batch, self.batch_size);
        }
        if pw_stride == 0 {
            anyhow::bail!("pw_stride must be > 0");
        }
        if pw_base.len() < pw_stride * batch {
            anyhow::bail!("pw_base too small for batch");
        }
        if pw_lens.len() < batch {
            anyhow::bail!("pw_lens too small for batch");
        }
        if out_hashes.len() < batch * 32 {
            anyhow::bail!("out_hashes too small for batch");
        }

        let rc = unsafe {
            argon2_gpu_hash_batch(
                self.job,
                pw_base.as_ptr(),
                pw_stride,
                pw_lens.as_ptr(),
                batch,
                out_hashes.as_mut_ptr(),
            )
        };
        if rc != 0 {
            anyhow::bail!("argon2_gpu_hash_batch failed with code {}", rc);
        }
        Ok(())
    }
}

impl Drop for GpuMiner {
    fn drop(&mut self) {
        if !self.job.is_null() {
            unsafe { argon2_gpu_job_destroy(self.job) };
            self.job = ptr::null_mut();
        }
    }
}
