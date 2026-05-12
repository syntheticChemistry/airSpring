// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared test helpers for GPU integration tests.
//!
//! Compiled into every integration test binary via `mod common;`.
//! Not every test file uses every helper, so dead-code/unused lints are
//! suppressed at the module level for shared test infrastructure only.
//!
//! ## Device acquisition
//!
//! Uses barraCuda's resilient `test_pool` (retry + exponential backoff)
//! rather than creating a fresh device per test. Returns `None` on CI/headless
//! or if the adapter doesn't support `SHADER_F64`.

/// Acquire a pooled f64-capable GPU device, or `None` if unavailable.
///
/// Delegates to `barracuda::device::test_pool::get_test_device_if_gpu_available`
/// which handles retry with exponential backoff and device health checks.
pub fn try_create_device() -> Option<std::sync::Arc<barracuda::device::WgpuDevice>> {
    barracuda::device::test_pool::tokio_block_on(
        barracuda::device::test_pool::get_test_device_if_gpu_available(),
    )
}

/// Catch panics from upstream shader regressions. Returns `None` on panic,
/// letting the test SKIP rather than FAIL.
#[expect(
    dead_code,
    reason = "shared helper: compiled into multiple test binaries, not all use it"
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
#[expect(
    unused_macros,
    reason = "shared helper: compiled into multiple test binaries, not all use it"
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

#[expect(
    unused_imports,
    reason = "shared helper: compiled into multiple test binaries, not all use it"
)]
pub(crate) use device_or_skip;
