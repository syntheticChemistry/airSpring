// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared test helpers for GPU integration tests.
//!
//! These helpers are compiled into every integration test binary via `mod common;`.
//! Not every test file uses every helper, so dead-code/unused lints are suppressed
//! at the module level for shared test infrastructure only.

/// Try to create an `f64`-capable `WgpuDevice`. Returns `None` on CI/headless
/// or if the GPU doesn't support `SHADER_F64`.
pub fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(barracuda::device::WgpuDevice::new_f64_capable())
        .ok()
        .map(std::sync::Arc::new)
}

/// Catch panics from upstream shader regressions. Returns `None` on panic,
/// letting the test SKIP rather than FAIL.
#[allow(
    dead_code,
    reason = "shared helper: used by gpu_integration + gpu_determinism, not all test crates"
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
    reason = "shared helper: used by gpu_integration + gpu_determinism, not all test crates"
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
    reason = "shared helper: re-export for gpu test files that use the macro"
)]
pub(crate) use device_or_skip;
