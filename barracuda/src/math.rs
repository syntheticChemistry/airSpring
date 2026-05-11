// SPDX-License-Identifier: AGPL-3.0-or-later
//! Math dispatch — delegates to barraCuda when `local` feature is enabled,
//! provides pure-Rust fallbacks when building IPC-only.
//!
//! Follows the ludoSpring V61 dual-path pattern for Tier 4 sovereignty.

/// Arithmetic mean of a slice.
#[inline]
pub fn mean(data: &[f64]) -> f64 {
    #[cfg(feature = "local")]
    {
        barracuda::stats::mean(data)
    }
    #[cfg(not(feature = "local"))]
    {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / crate::len_f64(data)
    }
}

/// Pearson correlation coefficient.
#[inline]
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(feature = "local")]
    {
        barracuda::stats::pearson_correlation(x, y).unwrap_or(f64::NAN)
    }
    #[cfg(not(feature = "local"))]
    {
        let n = crate::len_f64(x);
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            cov += dx * dy;
            vx += dx * dx;
            vy += dy * dy;
        }
        cov / (vx * vy).sqrt()
    }
}

/// Standard deviation (population).
#[inline]
pub fn std_dev(data: &[f64]) -> f64 {
    #[cfg(feature = "local")]
    {
        use barracuda::stats::correlation::std_dev as bc_std_dev;
        bc_std_dev(data).unwrap_or(f64::NAN)
    }
    #[cfg(not(feature = "local"))]
    {
        let m = mean(data);
        let var = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / crate::len_f64(data);
        var.sqrt()
    }
}
