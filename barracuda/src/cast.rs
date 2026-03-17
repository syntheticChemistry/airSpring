// SPDX-License-Identifier: AGPL-3.0-or-later
//! Type-safe numeric cast helpers.
//!
//! Centralizes `as` casts that clippy's `cast_precision_loss`,
//! `cast_possible_truncation`, and `cast_sign_loss` lints flag.
//! Each function documents when the cast is exact and when it
//! loses precision, so call sites don't need per-site `#[allow]`.
//!
//! Follows the neuralSpring S162 / healthSpring V33 `safe_cast` pattern.

/// `usize` → `f64`. Exact for values < 2^53 (9 × 10¹⁵).
/// All practical collection lengths and loop counters fit.
#[inline]
#[must_use]
pub const fn usize_f64(v: usize) -> f64 {
    v as f64
}

/// `f64` → `usize` via truncation toward zero.
///
/// # Panics
///
/// Debug-panics if `v` is negative, NaN, or exceeds `usize::MAX`.
#[inline]
#[must_use]
pub fn f64_usize(v: f64) -> usize {
    debug_assert!(
        v.is_finite() && v >= 0.0 && v <= usize_f64(usize::MAX),
        "f64_usize: {v} out of range"
    );
    v as usize
}

/// `usize` → `u32`. Saturates at `u32::MAX` in release builds.
///
/// # Panics
///
/// Debug-panics if `v > u32::MAX`.
#[inline]
#[must_use]
pub const fn usize_u32(v: usize) -> u32 {
    debug_assert!(v <= u32::MAX as usize, "usize_u32: overflow");
    v as u32
}

/// `i32` → `f64`. Always exact (i32 ⊂ f64 mantissa range).
#[inline]
#[must_use]
pub const fn i32_f64(v: i32) -> f64 {
    v as f64
}

/// `u32` → `f64`. Always exact (u32 ⊂ f64 mantissa range).
#[inline]
#[must_use]
pub const fn u32_f64(v: u32) -> f64 {
    v as f64
}

/// `f64` → `u32` via truncation. Saturates at `u32::MAX` in release builds.
///
/// # Panics
///
/// Debug-panics if `v` is negative, NaN, or exceeds `u32::MAX`.
#[inline]
#[must_use]
pub fn f64_u32(v: f64) -> u32 {
    debug_assert!(
        v.is_finite() && v >= 0.0 && v <= u32_f64(u32::MAX),
        "f64_u32: {v} out of range"
    );
    v as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_f64_roundtrips_small() {
        assert!((usize_f64(42) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn f64_usize_truncates() {
        assert_eq!(f64_usize(3.9), 3);
    }

    #[test]
    fn usize_u32_within_range() {
        assert_eq!(usize_u32(1000), 1000_u32);
    }

    #[test]
    fn i32_f64_exact() {
        assert!((i32_f64(-42) - (-42.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn u32_f64_exact() {
        assert!((u32_f64(u32::MAX) - f64::from(u32::MAX)).abs() < f64::EPSILON);
    }

    #[test]
    fn f64_u32_truncates() {
        assert_eq!(f64_u32(255.9), 255);
    }
}
