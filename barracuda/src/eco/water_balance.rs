//! Field-scale water balance model for irrigation scheduling.
//!
//! Implements the FAO-56 soil water balance (Allen et al., 1998, Ch. 8):
//!
//! ```text
//! Dr,i = Dr,i-1 - (P - RO)i - Ii - CRi + ETc,i + DPi
//! ```
//!
//! where:
//! - Dr = root zone depletion (mm)
//! - P = precipitation (mm)
//! - RO = runoff (mm)
//! - I = irrigation (mm)
//! - CR = capillary rise (mm)
//! - `ETc` = crop evapotranspiration (mm)
//! - DP = deep percolation (mm)

// ── Standalone water balance functions (match Python control API) ─────

/// Total Available Water (TAW) in the root zone (mm).
///
/// TAW = (θ\_FC − θ\_WP) × root\_depth
///
/// FAO-56 Eq. 82.
#[must_use]
pub fn total_available_water(theta_fc: f64, theta_wp: f64, root_depth_mm: f64) -> f64 {
    (theta_fc - theta_wp) * root_depth_mm
}

/// Readily Available Water (RAW) — the fraction of TAW easily extracted (mm).
///
/// RAW = p × TAW
///
/// FAO-56 Eq. 83. Typical p values: 0.5 for most crops.
#[must_use]
pub fn readily_available_water(taw: f64, p: f64) -> f64 {
    p * taw
}

/// Water stress coefficient Ks (standalone, FAO-56 Eq. 84).
///
/// Ks = (TAW − Dr) / (TAW − RAW) when Dr > RAW, else Ks = 1.0.
/// Returns 0.0 when Dr ≥ TAW.
#[must_use]
pub fn stress_coefficient(depletion: f64, taw: f64, raw: f64) -> f64 {
    if depletion <= raw {
        1.0
    } else if depletion >= taw {
        0.0
    } else {
        (taw - depletion) / (taw - raw)
    }
}

/// Daily water balance step (standalone, no state mutation).
///
/// Returns `(new_depletion, ks, actual_et, deep_percolation)`.
/// Matches the Python `daily_water_balance_step()` interface.
#[must_use]
pub fn daily_water_balance_step(
    dr_prev: f64,
    precipitation: f64,
    irrigation: f64,
    et0: f64,
    kc: f64,
    ks: f64,
    taw: f64,
) -> (f64, f64, f64) {
    let etc = et0 * kc;
    let actual_et = etc * ks;
    let mut new_dr = dr_prev - precipitation - irrigation + actual_et;
    let deep_percolation = if new_dr < 0.0 {
        let dp = -new_dr;
        new_dr = 0.0;
        dp
    } else {
        0.0
    };
    new_dr = new_dr.min(taw);
    (new_dr, actual_et, deep_percolation)
}

// ── Types ────────────────────────────────────────────────────────────

/// Runoff estimation method.
///
/// FAO-56 Ch. 8 defaults to no runoff for well-drained soils.
/// Alternative models can be added as capabilities here.
#[derive(Debug, Clone, Copy, Default)]
pub enum RunoffModel {
    /// No runoff — FAO-56 default for well-drained fields.
    /// Aligns with Python baseline (`control/water_balance/fao56_water_balance.py`).
    #[default]
    None,
    /// Simple threshold: `RO = (P − threshold) × fraction` when `P > threshold`.
    SimpleThreshold {
        /// Precipitation threshold before runoff begins (mm)
        threshold_mm: f64,
        /// Fraction of excess that becomes runoff (0–1)
        fraction: f64,
    },
}

impl RunoffModel {
    /// Compute runoff for a given precipitation amount.
    #[must_use]
    pub fn compute(&self, precipitation_mm: f64) -> f64 {
        match self {
            Self::None => 0.0,
            Self::SimpleThreshold {
                threshold_mm,
                fraction,
            } => {
                if precipitation_mm > *threshold_mm {
                    (precipitation_mm - threshold_mm) * fraction
                } else {
                    0.0
                }
            }
        }
    }
}

/// Daily water balance input.
#[derive(Debug, Clone)]
pub struct DailyInput {
    /// Precipitation (mm)
    pub precipitation: f64,
    /// Irrigation applied (mm)
    pub irrigation: f64,
    /// Reference ET₀ (mm/day)
    pub et0: f64,
    /// Crop coefficient Kc (dimensionless)
    pub kc: f64,
}

/// Water balance state for a single soil layer.
#[derive(Debug, Clone)]
pub struct WaterBalanceState {
    /// Root zone depletion Dr (mm) — 0 = at field capacity
    pub depletion: f64,
    /// Total available water TAW (mm)
    pub taw: f64,
    /// Readily available water RAW (mm)
    pub raw: f64,
    /// Field capacity (m³/m³)
    pub fc: f64,
    /// Wilting point (m³/m³)
    pub wp: f64,
    /// Root zone depth (mm)
    pub root_depth_mm: f64,
    /// Fraction of TAW that is readily available (p, typically 0.5)
    pub p: f64,
    /// Runoff estimation model
    pub runoff_model: RunoffModel,
}

/// Daily water balance output.
#[derive(Debug, Clone)]
pub struct DailyOutput {
    /// Root zone depletion at end of day (mm)
    pub depletion: f64,
    /// Crop ET (mm)
    pub etc: f64,
    /// Deep percolation below root zone (mm)
    pub deep_percolation: f64,
    /// Runoff (mm)
    pub runoff: f64,
    /// Water stress coefficient Ks (0–1, 1 = no stress)
    pub ks: f64,
    /// Actual ET (mm) — may be reduced by stress
    pub actual_et: f64,
    /// Whether irrigation is triggered
    pub needs_irrigation: bool,
}

impl WaterBalanceState {
    /// Create a new water balance starting at field capacity (Dr = 0).
    ///
    /// Uses `RunoffModel::None` (FAO-56 default) for runoff estimation.
    #[must_use]
    pub fn new(fc: f64, wp: f64, root_depth_mm: f64, p: f64) -> Self {
        let taw = (fc - wp) * root_depth_mm;
        let raw = p * taw;
        Self {
            depletion: 0.0,
            taw,
            raw,
            fc,
            wp,
            root_depth_mm,
            p,
            runoff_model: RunoffModel::default(),
        }
    }

    /// Create with a specific runoff model.
    #[must_use]
    pub const fn with_runoff_model(mut self, model: RunoffModel) -> Self {
        self.runoff_model = model;
        self
    }

    /// Compute water stress coefficient Ks (FAO-56 Eq. 84).
    ///
    /// Ks = (TAW − Dr) / (TAW − RAW) when Dr > RAW, else Ks = 1.0
    #[must_use]
    pub fn stress_coefficient(&self) -> f64 {
        if self.depletion <= self.raw {
            1.0
        } else if self.depletion >= self.taw {
            0.0
        } else {
            (self.taw - self.depletion) / (self.taw - self.raw)
        }
    }

    /// Step the water balance forward one day.
    pub fn step(&mut self, input: &DailyInput) -> DailyOutput {
        // Crop ET under no stress
        let etc = input.et0 * input.kc;

        // Water stress coefficient
        let ks = self.stress_coefficient();
        let actual_et = etc * ks;

        // Runoff estimation (capability-based, not hardcoded)
        let runoff = self.runoff_model.compute(input.precipitation);
        let effective_precip = input.precipitation - runoff;

        // Update depletion: Dr,i = Dr,i-1 − P_eff − I + ETa
        let mut new_depletion = self.depletion - effective_precip - input.irrigation + actual_et;

        // Deep percolation: if depletion goes negative, excess drains
        let deep_percolation = if new_depletion < 0.0 {
            let dp = -new_depletion;
            new_depletion = 0.0;
            dp
        } else {
            0.0
        };

        // Cap depletion at TAW (can't dry below wilting point)
        new_depletion = new_depletion.min(self.taw);

        self.depletion = new_depletion;

        // Irrigation trigger: need irrigation when Dr > RAW
        let needs_irrigation = self.depletion > self.raw;

        DailyOutput {
            depletion: self.depletion,
            etc,
            deep_percolation,
            runoff,
            ks,
            actual_et,
            needs_irrigation,
        }
    }

    /// Current volumetric water content.
    #[must_use]
    pub fn current_theta(&self) -> f64 {
        self.fc - self.depletion / self.root_depth_mm
    }
}

/// Run a full season water balance simulation.
#[must_use]
pub fn simulate_season(
    initial_state: &WaterBalanceState,
    daily_inputs: &[DailyInput],
) -> (WaterBalanceState, Vec<DailyOutput>) {
    let mut state = initial_state.clone();
    let mut outputs = Vec::with_capacity(daily_inputs.len());

    for input in daily_inputs {
        let output = state.step(input);
        outputs.push(output);
    }

    (state, outputs)
}

/// Mass balance check: sum(P + I) should equal sum(ET + DP + RO + ΔS).
///
/// Returns the absolute mass balance error (mm). Should be < 0.01 for
/// correct implementations.
#[must_use]
pub fn mass_balance_check(
    daily_inputs: &[DailyInput],
    outputs: &[DailyOutput],
    initial_depletion: f64,
    final_depletion: f64,
) -> f64 {
    let total_precip: f64 = daily_inputs.iter().map(|d| d.precipitation).sum();
    let total_irrig: f64 = daily_inputs.iter().map(|d| d.irrigation).sum();
    let total_et: f64 = outputs.iter().map(|d| d.actual_et).sum();
    let total_deep_perc: f64 = outputs.iter().map(|d| d.deep_percolation).sum();
    let total_runoff: f64 = outputs.iter().map(|d| d.runoff).sum();

    // Mass balance: P + I − ET − DP − RO = ΔS
    // where ΔS = Dr_initial − Dr_final
    let inflow = total_precip + total_irrig;
    let outflow = total_et + total_deep_perc + total_runoff;
    let storage_change = initial_depletion - final_depletion;

    (inflow - outflow - storage_change).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_stress_at_fc() {
        let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        assert!((state.stress_coefficient() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_full_stress_at_wp() {
        let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        state.depletion = state.taw;
        assert!((state.stress_coefficient()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stress_at_raw_boundary() {
        let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        state.depletion = state.raw;
        assert!((state.stress_coefficient() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stress_coefficient_midpoint() {
        let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        state.depletion = f64::midpoint(state.taw, state.raw);
        assert!((state.stress_coefficient() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mass_balance_dry_down() {
        let state = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..30)
            .map(|_| DailyInput {
                precipitation: 0.0,
                irrigation: 0.0,
                et0: 5.0,
                kc: 1.0,
            })
            .collect();
        let (final_state, outputs) = simulate_season(&state, &inputs);
        let error = mass_balance_check(&inputs, &outputs, state.depletion, final_state.depletion);
        assert!(error < 0.01, "Mass balance error: {error}");
    }

    #[test]
    fn test_mass_balance_irrigated() {
        let state = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..60)
            .map(|day| DailyInput {
                precipitation: if day % 7 == 3 { 15.0 } else { 0.0 },
                irrigation: if day % 10 == 0 { 25.0 } else { 0.0 },
                et0: 4.5,
                kc: 0.003f64.mul_add(f64::from(day), 0.85),
            })
            .collect();
        let (final_state, outputs) = simulate_season(&state, &inputs);
        let error = mass_balance_check(&inputs, &outputs, state.depletion, final_state.depletion);
        assert!(error < 0.01, "Mass balance error: {error}");
    }

    #[test]
    fn test_runoff_model_none() {
        let model = RunoffModel::None;
        assert!((model.compute(50.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_runoff_model_threshold() {
        let model = RunoffModel::SimpleThreshold {
            threshold_mm: 20.0,
            fraction: 0.2,
        };
        assert!((model.compute(10.0)).abs() < f64::EPSILON);
        assert!((model.compute(30.0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_current_theta_at_fc() {
        let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        assert!((state.current_theta() - 0.33).abs() < f64::EPSILON);
    }

    #[test]
    fn test_determinism() {
        // Same inputs must produce identical outputs on every run.
        let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        let inputs: Vec<DailyInput> = (0..30)
            .map(|_| DailyInput {
                precipitation: 5.0,
                irrigation: 0.0,
                et0: 4.0,
                kc: 1.0,
            })
            .collect();
        let (final1, out1) = simulate_season(&state, &inputs);
        let (final2, out2) = simulate_season(&state, &inputs);
        assert!((final1.depletion - final2.depletion).abs() < f64::EPSILON);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a.actual_et - b.actual_et).abs() < f64::EPSILON);
        }
    }
}
