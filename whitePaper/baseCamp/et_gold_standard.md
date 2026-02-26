# Direct ET Validation via AmeriFlux Eddy Covariance

**Date:** February 26, 2026
**Status:** Planning
**Data Tier:** 2 (free with registration)
**Dependencies:** Existing ET₀ module + AmeriFlux account
**Cross-Spring:** None (airSpring standalone)

---

## What This Is

Validate our FAO-56 ET₀ against eddy covariance (EC) measurements from
AmeriFlux/FLUXNET towers. EC is the gold standard for direct ET measurement —
it measures actual water vapor flux from the surface, not a reference
calculation like Open-Meteo.

## Why It Matters

Our current ET₀ validation (Exp 005, Phase 0+) cross-compares against
Open-Meteo's independent ET₀ calculation (R²=0.967). Both compute reference
ET₀ from weather data — they agree because they use similar methods.

EC towers measure *actual* evapotranspiration from the land surface. This
includes crop-specific transpiration, soil evaporation, and canopy interception.
Comparing our ET₀ (reference grass) × Kc (crop coefficient) against EC-measured
ETa for a specific crop provides a fundamentally different validation:
model vs measurement, not model vs model.

## Data Sources

| Source | Sites | Years | Variables | Access |
|--------|:-----:|:-----:|-----------|--------|
| AmeriFlux | ~500 US | 1990s-present | LE, H, NEE, Rn, Ta, RH, WS, P | Free (registration) |
| FLUXNET2015 | ~200 global | 1990s-2014 | Same + gap-filled | Free (registration) |

Michigan-relevant AmeriFlux sites:

| Site ID | Name | Land Cover | Years | Distance from our stations |
|---------|------|-----------|:-----:|---------------------------|
| US-KFS | Kellogg Forest Station | Deciduous forest | 2018+ | ~40km from MSU |
| US-Syv | Sylvania Wilderness | Mixed forest | 2001+ | Upper Peninsula |

Midwest agricultural sites:

| Site ID | Name | Land Cover | Years |
|---------|------|-----------|:-----:|
| US-Ne1/2/3 | Mead, Nebraska | Irrigated/rainfed corn-soy | 2001+ |
| US-Bo1/2 | Bondville, IL | Corn-soy rotation | 1996+ |
| US-IB1/2 | Fermi, IL | Corn-soy rotation | 2005+ |

## Methodology

1. Download AmeriFlux data for Michigan and Midwest agricultural sites
2. Compute FAO-56 ET₀ from tower meteorological data (Ta, RH, WS, Rn)
3. Apply crop-specific Kc (from our dual Kc module) to get predicted ETa
4. Compare predicted ETa against EC-measured LE/λ (latent heat → ET conversion)
5. Quantify: daily R², RMSE, seasonal bias, energy balance closure effects

## Expected Challenges

- **Energy balance closure**: EC towers typically close 80-90% of available energy.
  Corrected LE may be 10-20% higher than raw measurements.
- **Footprint mismatch**: EC measures a dynamic footprint (~100-500m), our model
  assumes a point. Heterogeneous landscapes introduce noise.
- **Reference vs actual**: FAO-56 ET₀ is reference grass; Kc adjustment is
  approximate. Direct comparison needs crop-specific Kc appropriate to site.

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `eco::evapotranspiration` | Exp 001/005/007 | 64+61+R²=0.967 |
| `eco::dual_kc` | Exp 005/011 | 103/103 |
| `eco::crop` | Exp 005 | 10 crops, Kc table |
