// SPDX-License-Identifier: AGPL-3.0-or-later
//! Green-Ampt (1911) infiltration model.
//!
//! Estimates cumulative infiltration F(t) and rate f(t) from soil properties:
//!
//! ```text
//! F(t) = Ks·t + ψ·Δθ·ln(1 + F(t)/(ψ·Δθ))   [implicit, solved iteratively]
//! f(t) = Ks·(1 + ψ·Δθ/F(t))                   [rate from cumulative]
//! ```
//!
//! # Ponding Time
//!
//! Under constant rainfall i > Ks:
//! tp = Ks·ψ·Δθ / (i·(i − Ks))
//!
//! # Reference
//!
//! Green WH, Ampt GA (1911) *Studies on Soil Physics.* J Agr Sci 4(1):1-24.
//! Rawls WJ, Brakensiek DL, Miller N (1983) *Green-Ampt parameters from soils
//! data.* J Hydraul Eng 109(1):62-70.

/// Green-Ampt soil hydraulic parameters (Rawls et al. 1983).
#[derive(Debug, Clone, Copy)]
pub struct GreenAmptParams {
    /// Saturated hydraulic conductivity (cm/hr).
    pub ks_cm_hr: f64,
    /// Wetting front suction head (cm, positive).
    pub psi_cm: f64,
    /// Moisture deficit: θs − θi.
    pub delta_theta: f64,
}

impl GreenAmptParams {
    /// Sand: Ks=11.78 cm/hr, ψ=4.95 cm.
    pub const SAND: Self = Self {
        ks_cm_hr: 11.78,
        psi_cm: 4.95,
        delta_theta: 0.417,
    };
    /// Loamy sand.
    pub const LOAMY_SAND: Self = Self {
        ks_cm_hr: 2.99,
        psi_cm: 6.13,
        delta_theta: 0.401,
    };
    /// Sandy loam.
    pub const SANDY_LOAM: Self = Self {
        ks_cm_hr: 1.09,
        psi_cm: 11.01,
        delta_theta: 0.412,
    };
    /// Loam.
    pub const LOAM: Self = Self {
        ks_cm_hr: 0.34,
        psi_cm: 8.89,
        delta_theta: 0.434,
    };
    /// Silt loam.
    pub const SILT_LOAM: Self = Self {
        ks_cm_hr: 0.65,
        psi_cm: 16.68,
        delta_theta: 0.486,
    };
    /// Clay loam.
    pub const CLAY_LOAM: Self = Self {
        ks_cm_hr: 0.10,
        psi_cm: 20.88,
        delta_theta: 0.309,
    };
    /// Clay.
    pub const CLAY: Self = Self {
        ks_cm_hr: 0.03,
        psi_cm: 31.63,
        delta_theta: 0.385,
    };
}

/// Cumulative infiltration F(t) via Newton-Raphson on the implicit GA equation.
///
/// Solves: F − Ks·t − ψ·Δθ·ln(1 + F/(ψ·Δθ)) = 0
///
/// Returns cumulative infiltration in cm.
#[must_use]
pub fn cumulative_infiltration(params: &GreenAmptParams, t_hr: f64) -> f64 {
    if t_hr <= 0.0 {
        return 0.0;
    }
    let ks = params.ks_cm_hr;
    let psi_dt = params.psi_cm * params.delta_theta;

    let mut f = ks.mul_add(t_hr, (2.0 * ks * psi_dt * t_hr).sqrt());

    for _ in 0..100 {
        if f <= 0.0 {
            f = ks * t_hr * 0.01;
        }
        let g = psi_dt.mul_add(-(f / psi_dt).ln_1p(), ks.mul_add(-t_hr, f));
        let dg = 1.0 - psi_dt / (psi_dt + f);
        if dg.abs() < 1e-15 {
            break;
        }
        let f_new = f - g / dg;
        let f_new = if f_new < 0.0 { f * 0.5 } else { f_new };
        if (f_new - f).abs() < 1e-10 {
            f = f_new;
            break;
        }
        f = f_new;
    }

    f.max(0.0)
}

/// Infiltration rate f(t) from cumulative F (cm/hr).
///
/// f = Ks × (1 + ψ·Δθ / F)
///
/// At F=0 the rate is theoretically infinite.
#[must_use]
pub fn infiltration_rate(params: &GreenAmptParams, f_cumulative_cm: f64) -> f64 {
    if f_cumulative_cm <= 0.0 {
        return f64::MAX;
    }
    params.ks_cm_hr * (1.0 + params.psi_cm * params.delta_theta / f_cumulative_cm)
}

/// Infiltration rate at time t (cm/hr). Convenience wrapper.
#[must_use]
pub fn infiltration_rate_at(params: &GreenAmptParams, t_hr: f64) -> f64 {
    let f = cumulative_infiltration(params, t_hr);
    infiltration_rate(params, f)
}

/// Time to ponding under constant rainfall intensity (hr).
///
/// tp = Ks·ψ·Δθ / (i·(i − Ks))
///
/// Returns `f64::INFINITY` if i ≤ Ks (ponding never occurs).
#[must_use]
pub fn ponding_time(params: &GreenAmptParams, rain_intensity_cm_hr: f64) -> f64 {
    let ks = params.ks_cm_hr;
    if rain_intensity_cm_hr <= ks {
        return f64::INFINITY;
    }
    ks * params.psi_cm * params.delta_theta / (rain_intensity_cm_hr * (rain_intensity_cm_hr - ks))
}

/// Infiltration time series: compute F(t) and f(t) at each time step.
///
/// Returns (`cumulative_cm`, `rate_cm_hr`) pairs.
#[must_use]
pub fn infiltration_series(params: &GreenAmptParams, times_hr: &[f64]) -> Vec<(f64, f64)> {
    times_hr
        .iter()
        .map(|&t| {
            let f_cum = cumulative_infiltration(params, t);
            let rate = infiltration_rate(params, f_cum);
            (f_cum, rate)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandy_loam_1hr() {
        let p = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let f = cumulative_infiltration(&p, 1.0);
        assert!((f - 3.51).abs() < 0.2, "F={f}");
        let rate = infiltration_rate(&p, f);
        assert!(rate > p.ks_cm_hr, "rate {rate} should exceed Ks");
    }

    #[test]
    fn sand_fast_infiltration() {
        let p = GreenAmptParams {
            delta_theta: 0.367,
            ..GreenAmptParams::SAND
        };
        let f = cumulative_infiltration(&p, 1.0);
        assert!(f > 10.0, "Sand should infiltrate fast: F={f}");
    }

    #[test]
    fn clay_slow_infiltration() {
        let p = GreenAmptParams {
            delta_theta: 0.285,
            ..GreenAmptParams::CLAY
        };
        let f = cumulative_infiltration(&p, 1.0);
        assert!(f < 2.0, "Clay should infiltrate slowly: F={f}");
    }

    #[test]
    fn zero_time_zero_infiltration() {
        let p = GreenAmptParams::SANDY_LOAM;
        assert!(cumulative_infiltration(&p, 0.0).abs() < 1e-12);
    }

    #[test]
    fn cumulative_monotonic() {
        let p = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let mut prev = 0.0;
        for t in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0] {
            let f = cumulative_infiltration(&p, t);
            assert!(f >= prev - 1e-10, "F not monotonic at t={t}");
            prev = f;
        }
    }

    #[test]
    fn rate_decreasing() {
        let p = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let mut prev_rate = f64::MAX;
        for t in [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0] {
            let rate = infiltration_rate_at(&p, t);
            assert!(rate <= prev_rate + 1e-10, "rate not decreasing at t={t}");
            prev_rate = rate;
        }
    }

    #[test]
    fn rate_bounded_below_by_ks() {
        let p = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        for t in [0.1, 1.0, 10.0, 100.0] {
            let rate = infiltration_rate_at(&p, t);
            assert!(rate >= p.ks_cm_hr - 1e-10, "rate {rate} < Ks at t={t}");
        }
    }

    #[test]
    fn asymptotic_to_ks() {
        let p = GreenAmptParams {
            delta_theta: 0.312,
            ..GreenAmptParams::SANDY_LOAM
        };
        let rate = infiltration_rate_at(&p, 100.0);
        assert!(
            (rate / p.ks_cm_hr - 1.0).abs() < 0.05,
            "ratio={}",
            rate / p.ks_cm_hr
        );
    }

    #[test]
    fn ponding_time_loam() {
        let p = GreenAmptParams {
            delta_theta: 0.405,
            ..GreenAmptParams::LOAM
        };
        let tp = ponding_time(&p, 2.0);
        assert!((tp - 0.37).abs() < 0.1, "tp={tp}");
    }

    #[test]
    fn no_ponding_when_i_leq_ks() {
        let p = GreenAmptParams::SAND;
        let tp = ponding_time(&p, 5.0); // i=5 < Ks=11.78
        assert!(tp.is_infinite(), "Should never pond: tp={tp}");
    }

    #[test]
    fn series_correct_length() {
        let p = GreenAmptParams::SANDY_LOAM;
        let times = vec![0.1, 0.5, 1.0, 2.0];
        let series = infiltration_series(&p, &times);
        assert_eq!(series.len(), 4);
        for &(f_cum, rate) in &series {
            assert!(f_cum >= 0.0);
            assert!(rate >= p.ks_cm_hr);
        }
    }

    #[test]
    fn named_constants_physical() {
        let soils = [
            GreenAmptParams::SAND,
            GreenAmptParams::LOAMY_SAND,
            GreenAmptParams::SANDY_LOAM,
            GreenAmptParams::LOAM,
            GreenAmptParams::SILT_LOAM,
            GreenAmptParams::CLAY_LOAM,
            GreenAmptParams::CLAY,
        ];
        for s in soils {
            assert!(s.ks_cm_hr > 0.0);
            assert!(s.psi_cm > 0.0);
            assert!(s.delta_theta > 0.0 && s.delta_theta < 1.0);
        }
    }
}
