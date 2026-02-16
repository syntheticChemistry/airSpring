//! airSpring BarraCUDA — Ecological & Agricultural Science Pipelines
//!
//! Rust implementations validated against FAO-56, HYDRUS, and published
//! field data from Dr. Younsuk Dong (MSU Biosystems & Agricultural Engineering).
//!
//! # Track 1: Precision Agriculture
//! - `eco::evapotranspiration` — FAO-56 Penman-Monteith reference ET₀
//! - `eco::soil_moisture` — Dielectric sensor calibration (Topp equation)
//! - `eco::water_balance` — Field-scale irrigation water budget
//!
//! # Track 2: Environmental Systems
//! - `eco::isotherms` — Adsorption isotherm fitting (Langmuir, Freundlich)
//! - `eco::richards` — Richards equation for unsaturated flow (future)
//!
//! # I/O
//! - `io::csv_ts` — Time series CSV parser for IoT sensor data

pub mod eco;
pub mod io;
