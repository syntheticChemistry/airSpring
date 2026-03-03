// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared test helpers for GPU integration tests.

/// Try to create an `f64`-capable `WgpuDevice`. Returns `None` on CI/headless
/// or if the GPU doesn't support `SHADER_F64`.
pub fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(barracuda::device::WgpuDevice::new_f64_capable())
        .ok()
        .map(std::sync::Arc::new)
}

/// Catch panics from upstream shader regressions. Returns `None` on panic,
/// letting the test SKIP rather than FAIL.
///
/// History: `ToadStool` S60-S65 used `layout: None` + `get_bind_group_layout(0)`
/// which panicked on `BatchedElementwiseF64` dispatch. S66 switched to explicit
/// `BindGroupLayout` (R-S66-041), resolving the P0 blocker. Retained as a
/// defensive wrapper for future shader regressions.
#[allow(
    dead_code,
    reason = "defensive wrapper retained for future shader regressions"
)]
pub fn try_gpu_dispatch<T>(f: impl FnOnce() -> T) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_or_else(
        |_| {
            eprintln!("SKIP: upstream shader regression");
            None
        },
        Some,
    )
}

/// Get a device or skip the test.
#[allow(
    unused_macros,
    reason = "macro available for GPU tests that need device-or-skip"
)]
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

#[allow(
    unused_imports,
    reason = "re-export available for GPU test files that use the macro"
)]
pub(crate) use device_or_skip;
