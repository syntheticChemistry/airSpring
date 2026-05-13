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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&data);
        assert!((m - 3.0).abs() < 1e-14, "mean={m}");
    }

    #[test]
    fn mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn pearson_r_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y);
        assert!((r - 1.0).abs() < 1e-12, "r={r}");
    }

    #[test]
    fn pearson_r_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_r(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-12, "r={r}");
    }

    #[test]
    fn std_dev_positive_for_varied_data() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!(sd > 1.9 && sd < 2.2, "std_dev={sd}");
    }

    #[test]
    fn std_dev_zero_for_constant() {
        let data = [5.0, 5.0, 5.0, 5.0];
        let sd = std_dev(&data);
        assert!(sd.abs() < 1e-12, "std_dev={sd}");
    }
}
