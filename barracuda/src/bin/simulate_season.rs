// SPDX-License-Identifier: AGPL-3.0-or-later
//! Simulate a complete growing season using the airSpring pipeline.
//!
//! Demonstrates the full workflow:
//! crop database → soil properties → daily ET₀ → water balance → scheduling
//!
//! Uses deterministic synthetic weather matching the Michigan summer scenario
//! from `control/water_balance/fao56_water_balance.py` so results can be
//! cross-validated.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --bin simulate_season
//! ```

use airspring_barracuda::eco::{
    crop::CropType,
    evapotranspiration::{self as et, DailyEt0Input},
    soil_moisture::SoilTexture,
    water_balance::{DailyInput, WaterBalanceState},
};

/// Guard against `ln(0)` in Box-Muller transform.
const LN_GUARD: f64 = 1e-15;

/// Probability of rain on any given day (Michigan summer climatology).
const RAIN_PROBABILITY: f64 = 0.30;

/// Mean rainfall per rainy event (mm), exponential distribution parameter.
const RAIN_MEAN_MM: f64 = 12.0;

/// Maximum rainfall cap per day (mm) to prevent unrealistic extremes.
const RAIN_CAP_MM: f64 = 80.0;

/// Maximum irrigation applied per event (mm). Reflects typical drip/sprinkler
/// capacity: enough to refill depleted root zone without runoff.
const MAX_IRRIGATION_MM: f64 = 25.0;

/// Simple deterministic pseudo-random number generator (Xorshift64).
/// Produces the same sequence on every platform — no external dependency.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    const fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Normal (Box-Muller, deterministic).
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(LN_GUARD);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z.mul_add(std, mean)
    }
}

/// Simulation results for one management strategy.
struct SimResult {
    total_et: f64,
    stress_days: u32,
    final_depletion: f64,
    total_irrigation: f64,
    irrigation_events: u32,
}

/// Run rainfed simulation (no irrigation).
fn simulate_rainfed(
    soil: &airspring_barracuda::eco::soil_moisture::SoilHydraulicProps,
    crop: &airspring_barracuda::eco::crop::CropCoefficients,
    root_zone_mm: f64,
    et0: &[f64],
    precip: &[f64],
) -> SimResult {
    let mut state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        root_zone_mm,
        crop.depletion_fraction,
    );
    let mut total_et = 0.0;
    let mut stress_days = 0u32;

    for (day_idx, (&e, &p)) in et0.iter().zip(precip).enumerate() {
        let _ = day_idx;
        let output = state.step(&DailyInput {
            precipitation: p,
            irrigation: 0.0,
            et0: e,
            kc: crop.kc_mid,
        });
        total_et += output.actual_et;
        if output.ks < 1.0 {
            stress_days += 1;
        }
    }

    SimResult {
        total_et,
        stress_days,
        final_depletion: state.depletion,
        total_irrigation: 0.0,
        irrigation_events: 0,
    }
}

/// Run smart-irrigation simulation (trigger at RAW).
fn simulate_smart(
    soil: &airspring_barracuda::eco::soil_moisture::SoilHydraulicProps,
    crop: &airspring_barracuda::eco::crop::CropCoefficients,
    root_zone_mm: f64,
    et0: &[f64],
    precip: &[f64],
) -> SimResult {
    let mut state = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        root_zone_mm,
        crop.depletion_fraction,
    );
    let raw = state.taw * crop.depletion_fraction;
    let mut total_et = 0.0;
    let mut total_irrig = 0.0;
    let mut irrig_events = 0u32;
    let mut stress_days = 0u32;

    for (&e, &p) in et0.iter().zip(precip) {
        let irrigation = if state.depletion > raw {
            let amount = state.depletion.min(MAX_IRRIGATION_MM);
            total_irrig += amount;
            irrig_events += 1;
            amount
        } else {
            0.0
        };

        let output = state.step(&DailyInput {
            precipitation: p,
            irrigation,
            et0: e,
            kc: crop.kc_mid,
        });
        total_et += output.actual_et;
        if output.ks < 1.0 {
            stress_days += 1;
        }
    }

    SimResult {
        total_et,
        stress_days,
        final_depletion: state.depletion,
        total_irrigation: total_irrig,
        irrigation_events: irrig_events,
    }
}

/// Generate deterministic daily weather series (ET₀ + precipitation).
fn generate_weather(
    n_days: usize,
    doy_start: u32,
    latitude_deg: f64,
    elevation_m: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>) {
    let mut et0_series = Vec::with_capacity(n_days);
    let mut precip_series = Vec::with_capacity(n_days);

    for day_idx in 0..n_days {
        let day_idx_u32 = u32::try_from(day_idx).expect("season length fits u32");
        let day_idx_f = day_idx as f64;
        let doy = doy_start + day_idx_u32;

        let t_base = 4.0f64.mul_add(
            ((day_idx_f - 45.0) * std::f64::consts::PI / 90.0).sin(),
            22.0,
        );
        let tmax = t_base + rng.normal(5.0, 1.5);
        let tmin = t_base - rng.normal(5.0, 1.5);

        let lat_rad = latitude_deg.to_radians();
        let ra = et::extraterrestrial_radiation(lat_rad, doy);
        let n_hours = et::daylight_hours(lat_rad, doy);

        let sunshine = n_hours * rng.normal(0.65, 0.10).clamp(0.2, 0.95);
        let rs = et::solar_radiation_from_sunshine(sunshine, n_hours, ra);

        let rh_min = rng.normal(55.0, 8.0).clamp(30.0, 80.0);
        let rh_max = rng.normal(85.0, 5.0).clamp(rh_min + 10.0, 100.0);
        let ea = et::actual_vapour_pressure_rh(tmin, tmax, rh_min, rh_max);
        let u2 = rng.normal(2.0, 0.5).max(0.5);

        let input = DailyEt0Input {
            tmin,
            tmax,
            tmean: None,
            solar_radiation: rs,
            wind_speed_2m: u2,
            actual_vapour_pressure: ea,
            elevation_m,
            latitude_deg,
            day_of_year: doy,
        };
        et0_series.push(et::daily_et0(&input).et0);

        let p = if rng.uniform() < RAIN_PROBABILITY {
            (-RAIN_MEAN_MM * rng.uniform().max(LN_GUARD).ln()).min(RAIN_CAP_MM)
        } else {
            0.0
        };
        precip_series.push(p);
    }

    (et0_series, precip_series)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring — Growing Season Simulation");
    println!("═══════════════════════════════════════════════════════════\n");

    let crop = CropType::Corn.coefficients();
    let soil = SoilTexture::SandyLoam.hydraulic_properties();
    let n_days: usize = env_usize("WB_SEASON_DAYS", 90);
    let n_days_u32 = u32::try_from(n_days).expect("season length fits u32");
    let n_days_f = n_days as f64;
    let latitude_deg: f64 = env_f64("WB_LATITUDE", 42.77);
    let elevation_m = env_f64("WB_ELEVATION", 256.0);
    let doy_start: u32 = env_u32("WB_DOY_START", 152);
    let seed: u64 = std::env::var("WB_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let location_name = std::env::var("WB_LOCATION").unwrap_or_else(|_| "Lansing MI".into());
    let root_zone_mm = crop.root_depth_m * 1000.0;

    println!("  Crop:      {} (Kc_mid = {:.2})", crop.name, crop.kc_mid);
    println!(
        "  Soil:      SandyLoam (FC={:.2}, WP={:.2})",
        soil.field_capacity, soil.wilting_point
    );
    println!("  Root zone: {root_zone_mm:.0} mm");
    println!(
        "  Season:    {n_days} days (DOY {doy_start}–{})",
        doy_start + n_days_u32 - 1
    );
    println!("  Location:  {location_name} ({latitude_deg}°N, {elevation_m}m)");
    println!("  Config:    WB_LATITUDE, WB_ELEVATION, WB_DOY_START,");
    println!("             WB_SEASON_DAYS, WB_SEED, WB_LOCATION\n");

    let mut rng = Rng::new(seed);
    let (et0_series, precip_series) =
        generate_weather(n_days, doy_start, latitude_deg, elevation_m, &mut rng);

    let rf = simulate_rainfed(&soil, &crop, root_zone_mm, &et0_series, &precip_series);
    let sm = simulate_smart(&soil, &crop, root_zone_mm, &et0_series, &precip_series);

    let total_precip: f64 = precip_series.iter().sum();
    let mean_et0: f64 = et0_series.iter().sum::<f64>() / n_days_f;

    println!("── Weather Summary ─────────────────────────────────");
    println!("  Mean ET₀:       {mean_et0:.2} mm/day");
    println!("  Total precip:   {total_precip:.0} mm");
    println!(
        "  Rain days:      {}",
        precip_series.iter().filter(|&&p| p > 0.0).count()
    );

    println!("\n── Rainfed (no irrigation) ─────────────────────────");
    println!("  Total ET:       {:.0} mm", rf.total_et);
    println!("  Stress days:    {}/{n_days}", rf.stress_days);
    println!("  Final depletion: {:.1} mm", rf.final_depletion);

    println!("\n── Smart irrigation (trigger at RAW) ───────────────");
    println!("  Total ET:       {:.0} mm", sm.total_et);
    println!("  Stress days:    {}/{n_days}", sm.stress_days);
    println!(
        "  Irrigation:     {:.0} mm ({} events)",
        sm.total_irrigation, sm.irrigation_events
    );
    println!("  Final depletion: {:.1} mm", sm.final_depletion);

    let water_savings_pct = if sm.total_irrigation > 0.0 {
        (1.0 - sm.total_irrigation / sm.total_et) * 100.0
    } else {
        100.0
    };
    println!("  Water savings:  {water_savings_pct:.0}% vs naive replacement");

    println!("\n── Verification ───────────────────────────────────");
    println!(
        "  [{}] Smart irrigation reduces stress days",
        if sm.stress_days < rf.stress_days {
            "OK"
        } else {
            "!!"
        }
    );
    println!(
        "  [{}] Smart irrigation increases ET",
        if sm.total_et > rf.total_et {
            "OK"
        } else {
            "!!"
        }
    );
    println!(
        "  [{}] Irrigation less than total ET (efficient)",
        if sm.total_irrigation < sm.total_et {
            "OK"
        } else {
            "!!"
        }
    );

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Simulation complete. All values deterministic (seed={seed}).");
    println!("═══════════════════════════════════════════════════════════");
}
