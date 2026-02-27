# Predictive Irrigation with Weather Forecasts

**Date:** February 26, 2026
**Status:** Validated (Exp 025 — 19/19 checks pass)
**Data Tier:** 2 (API key, already in testing-secrets/)
**Dependencies:** Existing scheduling pipeline (Exp 014) + OpenWeatherMap forecast API
**Cross-Spring:** None (airSpring standalone)

---

## What This Is

Extend the validated irrigation scheduling optimizer (Exp 014: Ali, Dong & Lavely
2024) with 5-day weather forecast integration. Instead of reactive scheduling
(irrigate when soil is dry), predict soil moisture 5 days ahead and schedule
proactively.

This is the bridge from validated reproduction to the Penny Irrigation vision:
sovereign, predictive, consumer-hardware irrigation scheduling.

## Why It Matters

Current scheduling (Exp 014) uses historical or simulated weather. A farmer
needs forecasts. The OpenWeatherMap 5-day forecast API is free (with key),
global, and already scripted in `scripts/download_enviroweather.py`.

The pipeline becomes:
```
Current soil state (sensor or model)
    + 5-day weather forecast (OpenWeatherMap)
    → FAO-56 ET₀ forecast
    → Water balance projection
    → Predicted depletion trajectory
    → Optimal irrigation timing and amount
    → "Irrigate 25mm on Thursday" (actionable recommendation)
```

## Data Requirements

| Source | What | Cost | Status |
|--------|------|------|--------|
| OpenWeatherMap 5-day | 3-hour forecast, global | Free (1000 calls/day) | API key in `testing-secrets/` |
| Current conditions | Last 7 days weather | Free | Open-Meteo or OWM |
| Soil state | Initial depletion from model or sensor | Free | Computed from recent weather |

No new data infrastructure needed. Existing scripts cover the API calls.

## Methodology

1. **Initialize soil state**: Run water balance from season start to today using
   Open-Meteo historical data
2. **Fetch 5-day forecast**: OpenWeatherMap API for the station location
3. **Project ET₀ forward**: Apply FAO-56 to each forecast timestep
4. **Project water balance**: Continue simulation 5 days into the future
5. **Find optimal intervention**: When does Dr exceed RAW? How much water?
6. **Update daily**: Re-forecast each morning with new weather data

## Validation Strategy

- **Hindcast**: Run the forecast scheduler on 2023 growing season data, using
  actual weather as "perfect forecast." Compare against Exp 014 optimal results.
- **Degraded forecast**: Add increasing noise to forecasts. How much forecast
  error can the scheduler tolerate before recommendations degrade?
- **Multi-strategy comparison**: Compare forecast-integrated scheduling against
  the 5 strategies from Exp 014 (rainfed, MAD 50/60/70%, growth-stage).

## Validated Modules Used

| Module | Experiment | Checks |
|--------|-----------|--------|
| `eco::evapotranspiration` | Exp 001 | 64/64 + 31/31 |
| `eco::water_balance` | Exp 004 | 18/18 + 13/13 |
| `eco::dual_kc` | Exp 005 | 63/63 + 61/61 |
| `eco::yield_response` | Exp 012 | 32/32 + 32/32 |
| Scheduling pipeline | Exp 014 | 25/25 + 28/28 |

## Connection to Penny Irrigation

This is the core computation for Phase 4. A Penny Irrigation device runs:
1. Read soil sensor ($200 SoilWatch 10)
2. Fetch 5-day forecast (free API)
3. Run this scheduling algorithm (BarraCuda CPU, < 1ms)
4. Display recommendation (irrigate X mm on day Y)

No cloud. No subscription. No vendor lock-in. The algorithm is AGPL-3.0.
The weather data is free. The sensor is $200. The compute is a $35 Raspberry Pi.
