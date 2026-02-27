# Stewart Yield Model vs USDA NASS Reality

**Date:** February 26, 2026
**Status:** Validated (Exp 024 — 40/40 checks pass)
**Data Tier:** 1 (free, downloadable CSV)
**Dependencies:** Existing yield response module + USDA NASS Quick Stats
**Cross-Spring:** None (airSpring standalone)

---

## What This Is

Compare our validated Stewart (1977) yield response predictions against actual
county-level crop yields from USDA NASS. This is the first real-world test of
the yield response module (Exp 012) — moving beyond paper benchmarks to measured
agricultural outcomes.

## Why It Matters

Exp 012 validated the Stewart model against FAO-56 Table 24 Ky factors and
showed the math is correct. But the model's utility depends on whether
it predicts actual yields from actual weather. USDA NASS provides county-level
yields for major crops going back decades — free, public, downloadable.

If the Stewart model, driven by Open-Meteo weather through our ET₀ + water
balance pipeline, can explain observed yield variation across Michigan counties
and years, we have a validated crop yield prediction system on consumer hardware.

## Data Requirements

| Source | What | Volume | Cost | Access |
|--------|------|--------|------|--------|
| USDA NASS Quick Stats | County-level yields (corn, soy, wheat, dry beans, sugar beets) | ~50K records | Free | https://quickstats.nass.usda.gov/ |
| Open-Meteo ERA5 | County centroid weather (one station per county) | ~2.5M records | Free | Existing scripts |
| USDA SSURGO | County-average soil properties (FC, WP, Ksat) | ~83 records | Free | Web Soil Survey |
| FAO-56 Table 24 | Ky factors per crop per growth stage | ~50 records | Free | Published |

Total: < 100MB. All free, no API keys needed for NASS Quick Stats download.

## Methodology

1. **Download USDA NASS yields** for 5 Michigan crops, 83 counties, 20+ years
2. **Download Open-Meteo weather** for each county centroid
3. **Run ET₀ → water balance → yield response** for each county-year
4. **Compare predicted relative yield** against NASS actual yield
5. **Quantify**: R², RMSE, county-level correlation, year-to-year variation explained

## Expected Outcomes

The Stewart model is intentionally simple (yield = f(water stress only)). We
expect:
- **R² ~ 0.3-0.5** for rainfed crops (water stress explains ~30-50% of yield variation)
- **Higher R² in dry years** (water stress dominant factor)
- **Lower R² in wet years** (other factors dominate: disease, nutrient, management)
- **Regional patterns**: Western Michigan (sandier soils) more water-limited than Thumb

Even partial predictive power validates the pipeline. The remaining variance
motivates extensions: nutrient stress, pest pressure, management factors.

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `eco::evapotranspiration` | Exp 001 | 64/64 + 31/31 |
| `eco::water_balance` | Exp 004 | 18/18 + 13/13 |
| `eco::yield_response` | Exp 012 | 32/32 + 32/32 |
| `eco::crop` | Exp 005 | 10 crops, Kc table |

## USDA NASS Quick Stats API

```
https://quickstats.nass.usda.gov/api/api_GET/
?key=<API_KEY>
&source_desc=SURVEY
&sector_desc=CROPS
&group_desc=FIELD CROPS
&commodity_desc=CORN
&statisticcat_desc=YIELD
&unit_desc=BU / ACRE
&state_alpha=MI
&agg_level_desc=COUNTY
&year__GE=2000
&format=CSV
```

Free API key from https://quickstats.nass.usda.gov/api/ (instant registration).
