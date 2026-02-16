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
    #[allow(clippy::cast_precision_loss)]
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Normal (Box-Muller, deterministic).
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z.mul_add(std, mean)
    }
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring — Michigan Growing Season Simulation");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Configuration ───────────────────────────────────────────────
    let crop = CropType::Corn.coefficients();
    let soil = SoilTexture::SandyLoam.hydraulic_properties();
    let n_days: usize = 90; // June–August growing season
    let latitude_deg: f64 = 42.77; // Lansing, MI
    let elevation_m = 256.0;
    let doy_start: u32 = 152; // June 1

    println!("  Crop:      {} (Kc_mid = {:.2})", crop.name, crop.kc_mid);
    println!(
        "  Soil:      SandyLoam (FC={:.2}, WP={:.2})",
        soil.field_capacity, soil.wilting_point
    );
    println!("  Root zone: {:.0} mm", crop.root_depth_m * 1000.0);
    println!(
        "  Season:    {n_days} days (DOY {doy_start}–{})",
        doy_start + n_days as u32 - 1
    );
    println!("  Location:  Lansing MI ({latitude_deg}°N, {elevation_m}m)\n");

    // ── Generate deterministic weather ──────────────────────────────
    let mut rng = Rng::new(42);

    // Compute ET₀ for each day using Hargreaves (simplified, no wind/humidity data)
    let mut et0_series = Vec::with_capacity(n_days);
    let mut precip_series = Vec::with_capacity(n_days);

    for day_idx in 0..n_days {
        let doy = doy_start + day_idx as u32;

        // Temperature: diurnal seasonal pattern + noise
        let t_base = 4.0f64.mul_add(
            ((day_idx as f64 - 45.0) * std::f64::consts::PI / 90.0).sin(),
            22.0,
        );
        let tmax = t_base + rng.normal(5.0, 1.5);
        let tmin = t_base - rng.normal(5.0, 1.5);

        // ET₀ from Penman-Monteith with estimated inputs
        let lat_rad = latitude_deg.to_radians();
        let ra = et::extraterrestrial_radiation(lat_rad, doy);
        let n_hours = et::daylight_hours(lat_rad, doy);

        // Sunshine hours: ~60-70% of possible
        let sunshine = n_hours * rng.normal(0.65, 0.10).clamp(0.2, 0.95);
        let rs = et::solar_radiation_from_sunshine(sunshine, n_hours, ra);

        // Humidity and wind (typical Michigan summer)
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
        let result = et::daily_et0(&input);
        et0_series.push(result.et0);

        // Precipitation: 30% chance of rain, exponential(12mm)
        let p = if rng.uniform() < 0.30 {
            (-12.0 * rng.uniform().max(1e-15).ln()).min(80.0) // exponential, capped
        } else {
            0.0
        };
        precip_series.push(p);
    }

    // ── Simulate: No irrigation (rainfed) ───────────────────────────
    let root_zone_mm = crop.root_depth_m * 1000.0;
    let mut state_rainfed = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        root_zone_mm,
        crop.depletion_fraction,
    );

    let mut total_et_rainfed = 0.0;
    let mut stress_days_rainfed = 0u32;
    for day_idx in 0..n_days {
        let output = state_rainfed.step(&DailyInput {
            precipitation: precip_series[day_idx],
            irrigation: 0.0,
            et0: et0_series[day_idx],
            kc: crop.kc_mid,
        });
        total_et_rainfed += output.actual_et;
        if output.ks < 1.0 {
            stress_days_rainfed += 1;
        }
    }

    // ── Simulate: Smart irrigation (trigger at RAW) ─────────────────
    let mut state_smart = WaterBalanceState::new(
        soil.field_capacity,
        soil.wilting_point,
        root_zone_mm,
        crop.depletion_fraction,
    );

    let raw = state_smart.taw * crop.depletion_fraction;
    let mut total_et_smart = 0.0;
    let mut total_irrig = 0.0;
    let mut irrig_events = 0u32;
    let mut stress_days_smart = 0u32;

    for day_idx in 0..n_days {
        // Irrigate when depletion exceeds RAW
        let irrigation = if state_smart.depletion > raw {
            let amount = state_smart.depletion.min(25.0);
            total_irrig += amount;
            irrig_events += 1;
            amount
        } else {
            0.0
        };

        let output = state_smart.step(&DailyInput {
            precipitation: precip_series[day_idx],
            irrigation,
            et0: et0_series[day_idx],
            kc: crop.kc_mid,
        });
        total_et_smart += output.actual_et;
        if output.ks < 1.0 {
            stress_days_smart += 1;
        }
    }

    // ── Results ─────────────────────────────────────────────────────
    let total_precip: f64 = precip_series.iter().sum();
    let mean_et0: f64 = et0_series.iter().sum::<f64>() / n_days as f64;

    println!("── Weather Summary ─────────────────────────────────");
    println!("  Mean ET₀:       {mean_et0:.2} mm/day");
    println!("  Total precip:   {total_precip:.0} mm");
    println!(
        "  Rain days:      {}",
        precip_series.iter().filter(|&&p| p > 0.0).count()
    );

    println!("\n── Rainfed (no irrigation) ─────────────────────────");
    println!("  Total ET:       {total_et_rainfed:.0} mm");
    println!("  Stress days:    {stress_days_rainfed}/{n_days}");
    println!("  Final depletion: {:.1} mm", state_rainfed.depletion);

    println!("\n── Smart irrigation (trigger at RAW) ───────────────");
    println!("  Total ET:       {total_et_smart:.0} mm");
    println!("  Stress days:    {stress_days_smart}/{n_days}");
    println!("  Irrigation:     {total_irrig:.0} mm ({irrig_events} events)");
    println!("  Final depletion: {:.1} mm", state_smart.depletion);

    let water_savings_pct = if total_irrig > 0.0 {
        let naive_irrig = total_et_smart; // naive = replace all ET
        (1.0 - total_irrig / naive_irrig) * 100.0
    } else {
        100.0
    };
    println!("  Water savings:  {water_savings_pct:.0}% vs naive replacement");

    // ── Verification ────────────────────────────────────────────────
    println!("\n── Verification ───────────────────────────────────");
    println!(
        "  [{}] Smart irrigation reduces stress days",
        if stress_days_smart < stress_days_rainfed {
            "OK"
        } else {
            "!!"
        }
    );
    println!(
        "  [{}] Smart irrigation increases ET",
        if total_et_smart > total_et_rainfed {
            "OK"
        } else {
            "!!"
        }
    );
    println!(
        "  [{}] Irrigation less than total ET (efficient)",
        if total_irrig < total_et_smart {
            "OK"
        } else {
            "!!"
        }
    );

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Simulation complete. All values deterministic (seed=42).");
    println!("═══════════════════════════════════════════════════════════");
}
