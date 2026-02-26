// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic data generators for testing parsers and validation binaries.
//!
//! Produces deterministic data with known statistical properties so that
//! computed results can be verified analytically.
//!
//! **Not for production data ingestion.**

use crate::io::csv_ts::TimeseriesData;

/// Generate a synthetic `IoT` sensor CSV dataset for validation and testing.
///
/// Produces deterministic data with known statistical properties:
/// - Temperature: diurnal cycle, mean = 25 °C, amplitude ±8 °C
/// - Soil moisture: slowly decreasing, range 0.10–0.40 m³/m³
/// - PAR: bell curve centered at solar noon, max ≈ 1800 µmol/m²/s
/// - Humidity: inverse of temperature, range 55–85%
///
/// # Analytical properties (for 7+ full days)
///
/// | Column | Mean | Min | Max |
/// |--------|------|-----|-----|
/// | `temperature` | 25.0 | 17.0 | 33.0 |
/// | `humidity` | 70.0 | 55.0 | 85.0 |
/// | `par` | ~464 | 0.0 | 1800.0 |
#[must_use]
pub fn generate_synthetic_iot_data(n_records: usize) -> TimeseriesData {
    use std::f64::consts::PI;

    let column_names = vec![
        "soil_moisture_1".to_string(),
        "soil_moisture_2".to_string(),
        "temperature".to_string(),
        "humidity".to_string(),
        "par".to_string(),
    ];
    let mut timestamps = Vec::with_capacity(n_records);
    let mut cols: Vec<Vec<f64>> = (0..5).map(|_| Vec::with_capacity(n_records)).collect();

    for i in 0..n_records {
        let hour = i % 24;
        let day = i / 24;
        timestamps.push(format!("2024-07-{:02}T{hour:02}:00:00", day + 1));

        let base_sm = 0.002f64.mul_add(-(i as f64), 0.28);
        let sm1 = (0.02f64.mul_add((i as f64).mul_add(0.1, 0.0).sin(), base_sm)).clamp(0.10, 0.40);
        let sm2 = (0.015f64.mul_add((i as f64).mul_add(0.1, 1.0).sin(), base_sm - 0.01))
            .clamp(0.10, 0.40);
        cols[0].push(sm1);
        cols[1].push(sm2);

        let temp = 8.0f64.mul_add(((hour as f64 - 14.0) * PI / 12.0).cos(), 25.0);
        cols[2].push(temp);

        let rh = (-15.0f64).mul_add(((hour as f64 - 14.0) * PI / 12.0).cos(), 70.0);
        cols[3].push(rh);

        let par = if (6..=20).contains(&hour) {
            1800.0 * (-(((hour as f64 - 13.0) / 3.5).powi(2))).exp()
        } else {
            0.0
        };
        cols[4].push(par);
    }

    TimeseriesData::new(column_names, timestamps, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_synthetic_iot_data_length() {
        let n = 48;
        let data = generate_synthetic_iot_data(n);
        assert_eq!(data.len(), n);
        assert_eq!(data.num_columns(), 5);
    }

    #[test]
    fn generate_synthetic_iot_data_temperature_range() {
        let data = generate_synthetic_iot_data(168);
        let temps = data.column("temperature").unwrap();
        for &t in temps {
            assert!((17.0..=33.0).contains(&t), "temp {t} out of [17,33]");
        }
    }

    #[test]
    fn generate_synthetic_iot_data_humidity_range() {
        let data = generate_synthetic_iot_data(168);
        let rh = data.column("humidity").unwrap();
        for &h in rh {
            assert!((55.0..=85.0).contains(&h), "humidity {h} out of [55,85]");
        }
    }

    #[test]
    fn generate_synthetic_iot_data_soil_moisture_range() {
        let data = generate_synthetic_iot_data(168);
        let sm1 = data.column("soil_moisture_1").unwrap();
        let sm2 = data.column("soil_moisture_2").unwrap();
        for &v in sm1.iter().chain(sm2) {
            assert!(
                (0.10..=0.40).contains(&v),
                "soil moisture {v} out of [0.10,0.40]"
            );
        }
    }

    #[test]
    fn generate_synthetic_iot_data_par_range() {
        let data = generate_synthetic_iot_data(168);
        let par = data.column("par").unwrap();
        for &p in par {
            assert!((0.0..=1800.0).contains(&p), "PAR {p} out of [0,1800]");
        }
    }

    #[test]
    fn generate_synthetic_iot_data_deterministic() {
        let d1 = generate_synthetic_iot_data(24);
        let d2 = generate_synthetic_iot_data(24);
        let t1 = d1.column("temperature").unwrap();
        let t2 = d2.column("temperature").unwrap();
        for (a, b) in t1.iter().zip(t2) {
            assert!((a - b).abs() < 1e-10, "deterministic: {a} vs {b}");
        }
    }
}
