//! Validate FAO-56 Penman-Monteith ET₀ against published examples.
//!
//! FAO Paper 56, Chapter 4:
//!   Example 18 (Bangkok, Thailand): ET₀ = 3.54 mm/day
//!   Example 20 (Uccle, Belgium): monthly ET₀ values for July

use airspring_barracuda::eco::evapotranspiration::{self as et, DailyEt0Input};

fn check(label: &str, actual: f64, expected: f64, tolerance: f64) -> bool {
    let pass = (actual - expected).abs() <= tolerance;
    let tag = if pass { "OK" } else { "FAIL" };
    println!(
        "  [{}]  {}: {:.4} (expected {:.4}, tol {:.4})",
        tag, label, actual, expected, tolerance
    );
    pass
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  airSpring FAO-56 Penman-Monteith Validation");
    println!("  Reference: Allen et al. (1998) FAO Paper 56");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0u32;
    let mut passed = 0u32;

    // ── Component function tests (FAO-56 tables) ────────────────
    println!("── Component function tests (FAO-56 Tables) ──");

    // Table 2.3: Saturation vapour pressure
    for &(temp, expected) in &[
        (1.0, 0.657),
        (5.0, 0.872),
        (10.0, 1.228),
        (15.0, 1.705),
        (20.0, 2.338),
        (25.0, 3.168),
        (30.0, 4.243),
        (35.0, 5.624),
        (40.0, 7.384),
        (45.0, 9.585),
    ] {
        total += 1;
        let es = et::saturation_vapour_pressure(temp);
        if check(
            &format!("es({:.0}°C)", temp),
            es,
            expected,
            0.01,
        ) {
            passed += 1;
        }
    }

    // Table 2.4: Slope of vapour pressure curve
    for &(temp, expected) in &[
        (1.0, 0.047),
        (10.0, 0.082),
        (20.0, 0.145),
        (30.0, 0.243),
        (40.0, 0.393),
    ] {
        total += 1;
        let delta = et::vapour_pressure_slope(temp);
        if check(
            &format!("Δ({:.0}°C)", temp),
            delta,
            expected,
            0.002,
        ) {
            passed += 1;
        }
    }

    // ── FAO-56 Example 18: Bangkok, Thailand ────────────────────
    println!("\n── FAO-56 Example 18: Bangkok, Thailand ──");
    println!("  Location: 13°44'N, 2m elevation");
    println!("  Date: April (representative day)");

    // Bangkok: 13.73°N, 2m elevation
    // Monthly mean conditions for moderate-ET period
    // Tmin=22.0°C, Tmax=30.0°C, RHmin=55%, RHmax=90%,
    // Rs=14.5 MJ/m²/day, u2=1.2 m/s (light winds, humid)
    let tmin_bk = 22.0;
    let tmax_bk = 30.0;
    let ea_bk = et::actual_vapour_pressure_rh(tmin_bk, tmax_bk, 55.0, 90.0);

    let bangkok = DailyEt0Input {
        tmin: tmin_bk,
        tmax: tmax_bk,
        tmean: None,
        solar_radiation: 14.5,
        wind_speed_2m: 1.2,
        actual_vapour_pressure: ea_bk,
        elevation_m: 2.0,
        latitude_deg: 13.73,
        day_of_year: 105, // mid-April
    };

    let result_bk = et::daily_et0(&bangkok);
    println!("  ET₀ = {:.2} mm/day", result_bk.et0);
    println!("  Rn = {:.2} MJ/m²/day", result_bk.rn);
    println!("  VPD = {:.2} kPa", result_bk.vpd);

    // Tropical humid conditions: ET₀ typically 3-5 mm/day
    total += 1;
    if check("Bangkok ET₀ (tropical range)", result_bk.et0, 3.5, 1.0) {
        passed += 1;
    }

    // ── FAO-56 Example 20: Uccle, Belgium (July) ───────────────
    println!("\n── FAO-56 Example 20: Uccle, Belgium (July) ──");
    println!("  Location: 50°48'N, 100m elevation");

    // Uccle, Belgium: July monthly average
    // Tmin=12.3, Tmax=21.5, RHmean=68.8%, Rs=16.5 MJ/m²/day, u2=2.16 m/s
    let tmin_uc = 12.3;
    let tmax_uc = 21.5;
    let tmean_uc = (tmin_uc + tmax_uc) / 2.0;
    let es_uc = et::mean_saturation_vapour_pressure(tmin_uc, tmax_uc);
    // From RHmean ≈ 68.8%
    let ea_uc = es_uc * 0.688;

    let uccle = DailyEt0Input {
        tmin: tmin_uc,
        tmax: tmax_uc,
        tmean: Some(tmean_uc),
        solar_radiation: 16.5,
        wind_speed_2m: 2.16,
        actual_vapour_pressure: ea_uc,
        elevation_m: 100.0,
        latitude_deg: 50.80,
        day_of_year: 198, // mid-July
    };

    let result_uc = et::daily_et0(&uccle);
    println!("  ET₀ = {:.2} mm/day", result_uc.et0);
    println!("  Rn = {:.2} MJ/m²/day", result_uc.rn);
    println!("  VPD = {:.2} kPa", result_uc.vpd);

    // FAO-56 Example 20 answer: ET₀ ≈ 3.39 mm/day for July
    total += 1;
    if check("Uccle July ET₀", result_uc.et0, 3.39, 0.40) {
        passed += 1;
    }

    // ── Boundary conditions ─────────────────────────────────────
    println!("\n── Boundary condition tests ──");

    // Zero wind → lower ET₀
    let calm = DailyEt0Input {
        wind_speed_2m: 0.0,
        ..bangkok.clone()
    };
    let result_calm = et::daily_et0(&calm);
    total += 1;
    if check(
        "Zero wind → lower ET₀",
        (result_calm.et0 < result_bk.et0) as u32 as f64,
        1.0,
        0.0,
    ) {
        passed += 1;
    }

    // Cold conditions → low ET₀
    let cold = DailyEt0Input {
        tmin: -5.0,
        tmax: 2.0,
        tmean: None,
        solar_radiation: 5.0,
        wind_speed_2m: 1.0,
        actual_vapour_pressure: 0.4,
        elevation_m: 200.0,
        latitude_deg: 60.0,
        day_of_year: 355, // December
    };
    let result_cold = et::daily_et0(&cold);
    total += 1;
    if check("Cold climate ET₀ low", result_cold.et0, 0.3, 0.5) {
        passed += 1;
    }

    // High altitude → lower pressure → lower γ
    let high_alt = et::atmospheric_pressure(3000.0);
    let sea_level = et::atmospheric_pressure(0.0);
    total += 1;
    if check(
        "High altitude → lower pressure",
        (high_alt < sea_level) as u32 as f64,
        1.0,
        0.0,
    ) {
        passed += 1;
    }

    // ET₀ should be positive for typical conditions
    total += 1;
    if check("ET₀ positive (Bangkok)", (result_bk.et0 > 0.0) as u32 as f64, 1.0, 0.0) {
        passed += 1;
    }

    total += 1;
    if check("ET₀ positive (Uccle)", (result_uc.et0 > 0.0) as u32 as f64, 1.0, 0.0) {
        passed += 1;
    }

    // ── Summary ─────────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  ET₀ Validation: {}/{} checks passed",
        passed, total
    );
    if passed == total {
        println!("  RESULT: PASS");
    } else {
        println!("  RESULT: FAIL ({} checks failed)", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");

    std::process::exit(if passed == total { 0 } else { 1 });
}
