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
//! - ETc = crop evapotranspiration (mm)
//! - DP = deep percolation (mm)

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
    /// Water stress coefficient Ks (0-1, 1 = no stress)
    pub ks: f64,
    /// Actual ET (mm) — may be reduced by stress
    pub actual_et: f64,
    /// Whether irrigation is triggered
    pub needs_irrigation: bool,
}

impl WaterBalanceState {
    /// Create a new water balance starting at field capacity (Dr=0).
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
        }
    }

    /// Compute water stress coefficient Ks (FAO-56 Eq. 84).
    /// Ks = (TAW - Dr) / (TAW - RAW) when Dr > RAW, else Ks = 1.0
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

        // Effective precipitation (simple: assume no runoff for P < 20mm)
        let runoff = if input.precipitation > 20.0 {
            (input.precipitation - 20.0) * 0.2 // simple 20% runoff above 20mm
        } else {
            0.0
        };
        let effective_precip = input.precipitation - runoff;

        // Update depletion
        // Dr,i = Dr,i-1 - P_eff - I + ETa
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
    pub fn current_theta(&self) -> f64 {
        self.fc - self.depletion / self.root_depth_mm
    }
}

/// Run a full season water balance simulation.
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
pub fn mass_balance_check(
    daily_inputs: &[DailyInput],
    outputs: &[DailyOutput],
    initial_depletion: f64,
    final_depletion: f64,
) -> f64 {
    let total_p: f64 = daily_inputs.iter().map(|d| d.precipitation).sum();
    let total_i: f64 = daily_inputs.iter().map(|d| d.irrigation).sum();
    let total_et: f64 = outputs.iter().map(|d| d.actual_et).sum();
    let total_dp: f64 = outputs.iter().map(|d| d.deep_percolation).sum();
    let total_ro: f64 = outputs.iter().map(|d| d.runoff).sum();

    // Mass balance: P + I - ET - DP - RO = ΔS
    // where ΔS = Dr_initial - Dr_final (positive = storage increased = depletion decreased)
    let inflow = total_p + total_i;
    let outflow = total_et + total_dp + total_ro;
    let storage_change = initial_depletion - final_depletion;

    (inflow - outflow - storage_change).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_stress_at_fc() {
        let state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        assert_eq!(state.stress_coefficient(), 1.0);
    }

    #[test]
    fn test_full_stress_at_wp() {
        let mut state = WaterBalanceState::new(0.33, 0.13, 600.0, 0.5);
        state.depletion = state.taw;
        assert_eq!(state.stress_coefficient(), 0.0);
    }

    #[test]
    fn test_mass_balance() {
        let state = WaterBalanceState::new(0.30, 0.10, 500.0, 0.5);
        let inputs: Vec<DailyInput> = (0..30)
            .map(|_| DailyInput {
                precipitation: 2.0,
                irrigation: 0.0,
                et0: 4.0,
                kc: 1.0,
            })
            .collect();
        let (final_state, outputs) = simulate_season(&state, &inputs);
        let error = mass_balance_check(&inputs, &outputs, state.depletion, final_state.depletion);
        assert!(error < 0.01, "Mass balance error: {}", error);
    }
}
