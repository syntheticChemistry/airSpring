# Richards theta(t) vs USDA SCAN In-Situ Measurements

**Date:** February 26, 2026
**Status:** Validated (Exp 026 — 34/34 checks pass)
**Data Tier:** 2 (free, public API)
**Dependencies:** Existing Richards module (Exp 006/013) + USDA SCAN data
**Cross-Spring:** None (airSpring standalone), but theta(t) feeds baseCamp 06 (no-till Anderson)

---

## What This Is

Validate our Richards equation solver (`eco::richards`, `eco::van_genuchten`)
against in-situ soil moisture measurements from USDA SCAN (Soil Climate Analysis
Network) stations. This is the first real-data test of the PDE solver.

## Why It Matters

Exp 006 validated the Richards equation against van Genuchten (1980) analytical
solutions and Carsel & Parrish (1988) soil parameters. Exp 013 extended to CW2D
constructed wetland media. Both used published parameters, not measured field data.

USDA SCAN stations measure soil moisture at multiple depths (5, 10, 20, 50, 100 cm)
every hour, alongside weather data. This provides:
- **Boundary conditions**: Precipitation and ET from weather sensors
- **Ground truth**: Measured theta(t) at known depths
- **Soil properties**: Station metadata includes USDA soil texture

Running our Richards solver with SCAN weather as boundary conditions and comparing
predicted theta(z,t) against measured theta(z,t) is a genuine validation of the
PDE solver on real soil profiles.

## Data Sources

| Source | What | Volume | Cost | Access |
|--------|------|--------|------|--------|
| USDA SCAN | Hourly soil moisture + weather | ~2M records | Free | https://wcc.sc.egov.usda.gov/nwcc/inventory |
| USDA SSURGO | Soil hydraulic properties per site | ~50 records | Free | Web Soil Survey |

Michigan SCAN stations:

| Station | Location | Soil | Years | Depths |
|---------|----------|------|:-----:|--------|
| Kellogg | Hickory Corners, MI | Kalamazoo loam | 2000+ | 5, 10, 20, 50, 100 cm |
| Several others | Various MI locations | Various | Various | Multiple depths |

## Methodology

1. Download SCAN hourly data for Michigan stations (precipitation, air temp, soil temp, soil moisture at 5 depths)
2. Look up SSURGO soil properties (theta_r, theta_s, alpha, n, Ks) for each station
3. Construct van Genuchten parameter sets from SSURGO
4. Run Richards 1D solver with:
   - Upper BC: precipitation - ET₀ (computed from SCAN weather)
   - Lower BC: free drainage
   - Initial condition: measured theta profile at start
5. Compare predicted theta(z=5,10,20,50,100cm, t) against SCAN measurements
6. Quantify: RMSE, NSE, R² at each depth, seasonal performance

## Expected Challenges

- **Preferential flow**: Richards equation assumes matrix flow; macropore flow
  after heavy rain will cause model-measurement divergence
- **Root water uptake**: Simple sink term approximation vs actual root distribution
- **Frozen soil**: Richards equation needs modification for freeze-thaw cycles
- **Spatial heterogeneity**: SCAN point measurement vs model homogeneous column

## Connection to baseCamp 06 (No-Till Anderson)

The validated theta(t) is the input to the Anderson QS coupling:

```
SCAN weather → Richards solver → theta(z,t) → pore_connectivity(t)
    → d_eff(t) → Anderson r(t) → QS_regime(t)
```

If theta(t) is wrong, the entire Anderson-QS pipeline is wrong. This validation
is a prerequisite for baseCamp 06.

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `eco::richards` | Exp 006 | 14/14 + 15/15 |
| `eco::van_genuchten` | Exp 006/013 | Extracted module, 150 lines |
| `eco::evapotranspiration` | Exp 001 | 64/64 + 31/31 |
| `gpu::richards` | GPU wired | BatchedRichards via ToadStool S40 |

## Compute Requirements

| Scale | CPU Time | GPU Time |
|-------|----------|----------|
| 1 station, 1 year, hourly | ~10 sec | ~0.5 sec |
| 1 station, 20 years, hourly | ~3 min | ~10 sec |
| 10 stations, 20 years, hourly | ~30 min | ~2 min |

Richards at hourly timesteps is more expensive than daily water balance,
but still well within Eastgate's capacity. GPU batching helps at scale.
