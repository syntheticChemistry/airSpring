// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stateful multi-day dual Kc simulation.

use super::equations::{
    etc_dual, evaporation_layer_balance, evaporation_reduction, soil_evaporation_ke,
};
use super::mulch::mulched_ke;
use super::types::{DualKcInput, DualKcOutput, EvaporationLayerState};

/// Run a multi-day dual Kc simulation.
///
/// Returns per-day outputs and the final evaporation layer state.
#[must_use]
pub fn simulate_dual_kc(
    inputs: &[DualKcInput],
    kcb: f64,
    kc_max_val: f64,
    few: f64,
    state: &EvaporationLayerState,
) -> (Vec<DualKcOutput>, EvaporationLayerState) {
    let mut de = state.de;
    let tew = state.tew;
    let rew = state.rew;
    let mut outputs = Vec::with_capacity(inputs.len());

    for inp in inputs {
        de = (de - inp.precipitation - inp.irrigation).clamp(0.0, tew);

        let kr = evaporation_reduction(tew, rew, de);
        let ke = soil_evaporation_ke(kr, kcb, kc_max_val, few);
        let etc = etc_dual(kcb, 1.0, ke, inp.et0);

        outputs.push(DualKcOutput { de, kr, ke, etc });

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, inp.et0, few, tew);
    }

    (outputs, EvaporationLayerState { de, tew, rew })
}

/// Run a multi-day dual Kc simulation with mulch reduction on Ke.
///
/// Identical to [`simulate_dual_kc`] but applies `mulch_factor` to reduce
/// soil evaporation, modeling no-till residue effects.
#[must_use]
pub fn simulate_dual_kc_mulched(
    inputs: &[DualKcInput],
    kcb: f64,
    kc_max_val: f64,
    few: f64,
    mulch_factor: f64,
    state: &EvaporationLayerState,
) -> (Vec<DualKcOutput>, EvaporationLayerState) {
    let mut de = state.de;
    let tew = state.tew;
    let rew = state.rew;
    let mut outputs = Vec::with_capacity(inputs.len());

    for inp in inputs {
        de = (de - inp.precipitation - inp.irrigation).clamp(0.0, tew);

        let kr = evaporation_reduction(tew, rew, de);
        let ke = mulched_ke(kr, kcb, kc_max_val, few, mulch_factor);
        let etc = etc_dual(kcb, 1.0, ke, inp.et0);

        outputs.push(DualKcOutput { de, kr, ke, etc });

        de = evaporation_layer_balance(de, 0.0, 0.0, ke, inp.et0, few, tew);
    }

    (outputs, EvaporationLayerState { de, tew, rew })
}
