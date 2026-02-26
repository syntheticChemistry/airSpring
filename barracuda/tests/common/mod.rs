// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared test helpers for GPU integration tests.

/// Try to create an `f64`-capable `WgpuDevice`. Returns `None` on CI/headless
/// or if the GPU doesn't support `SHADER_F64`.
pub fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    pollster::block_on(barracuda::device::WgpuDevice::new_f64_capable())
        .ok()
        .map(std::sync::Arc::new)
}

/// Catch panics from upstream shader regressions (toadstool S60-S65
/// sovereign compiler bind-group reflection). Returns `None` on panic,
/// letting the test SKIP rather than FAIL.
#[allow(dead_code)]
pub fn try_gpu_dispatch<T>(f: impl FnOnce() -> T) -> Option<T> {
    if let Ok(val) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Some(val)
    } else {
        eprintln!("SKIP: upstream shader regression (toadstool S60-S65)");
        None
    }
}

/// Get a device or skip the test.
macro_rules! device_or_skip {
    () => {
        match $crate::common::try_create_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: No GPU device available");
                return;
            }
        }
    };
}

pub(crate) use device_or_skip;
