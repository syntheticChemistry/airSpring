// SPDX-License-Identifier: AGPL-3.0-or-later
//! IEEE-754–aware numeric guards (division, logarithms, probability transforms).
//!
//! Values are chosen well above [`f64::EPSILON`] (~2.2×10⁻¹⁶) so rounding does not
//! erase the guard in multiply-add chains, yet small enough to negligibly bias
//! physical quantities in agricultural and hydrologic units.
//!
//! IEEE 754 binary64 provides ~15 decimal digits of precision; exponents here sit
//! in the normal range (not subnormal) for stable library use.

/// Strictly-positive floor for probabilities, concentrations, and generic “small
/// but nonzero” clamps before `ln`, `norm_ppf`, power laws, etc.
pub const POSITIVE_DATA_GUARD: f64 = 1e-10;

/// Denominator / derivative floor to avoid division by zero while staying near
/// machine precision for [`f64`].
pub const DIVISION_GUARD: f64 = 1e-15;

/// Minimum uniform variate in `(0, 1]` before `ln(u)` (e.g. Box–Muller).
/// Well above [`f64::MIN_POSITIVE`] and in the normal range so `ln(u)` is finite
/// without depending on subnormal behavior.
pub const LOG_UNIFORM_FLOOR: f64 = 1e-300;

/// Treat regression slopes, normal-equation denominators, and similar scalars
/// below this magnitude as numerically singular.
pub const LINEAR_SYSTEM_EPSILON: f64 = 1e-30;
