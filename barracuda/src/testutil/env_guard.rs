// SPDX-License-Identifier: AGPL-3.0-or-later
//! RAII guard for environment variable manipulation in tests.
//!
//! Consolidates all `unsafe { env::set_var / env::remove_var }` usage into a
//! single utility. Rust 2024 marks these functions `unsafe` because they are
//! unsound in multi-threaded programs; our tests serialize env-touching tests
//! via `#[serial]`, making the safety invariant trivially satisfied.
//!
//! The guard saves the original value on creation and restores it on drop,
//! preventing inter-test pollution even if a test panics.

/// RAII guard that restores an environment variable to its original state on drop.
///
/// # Examples
///
/// ```ignore
/// let _g = EnvGuard::remove("MY_VAR");   // unsets MY_VAR, restores on drop
/// let _g = EnvGuard::set("MY_VAR", "x"); // sets MY_VAR=x, restores on drop
/// ```
pub struct EnvGuard {
    key: String,
    original: Option<String>,
}

impl EnvGuard {
    /// Remove an env var for the duration of the guard's lifetime.
    #[must_use]
    pub fn remove(key: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: test-only, serialized via `#[serial]` — no concurrent env mutation.
        unsafe {
            std::env::remove_var(key);
        }
        Self {
            key: key.to_owned(),
            original,
        }
    }

    /// Set an env var for the duration of the guard's lifetime.
    #[must_use]
    pub fn set(key: &str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: test-only, serialized via `#[serial]` — no concurrent env mutation.
        unsafe {
            std::env::set_var(key, value);
        }
        Self {
            key: key.to_owned(),
            original,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: restoring the original value — same serialized test context.
        unsafe {
            if let Some(ref val) = self.original {
                std::env::set_var(&self.key, val);
            } else {
                std::env::remove_var(&self.key);
            }
        }
    }
}
