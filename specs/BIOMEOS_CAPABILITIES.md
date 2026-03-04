# airSpring — biomeOS Capability Registration

**Updated**: March 2, 2026
**Status**: Active — ecology domain registered, 30 capabilities implemented (v0.6.9)
**Requires**: biomeOS Tower Node (stable), ToadStool (compute), NestGate (storage)

---

## Capability Domain: ecology

airSpring registers as an **ecology** domain provider, parallel to wetSpring's
**science** domain. Ecological capabilities are high-level orchestration endpoints
that internally compose `barracuda` CPU functions and ToadStool GPU shaders.

### Proposed Capability Registry Additions

```toml
[domains.ecology]
provider = "airspring"
capabilities = [
    "ecology",
    "irrigation",
    "soil_moisture",
    "evapotranspiration",
    "crop_science"
]

[translations.ecology]
# Semantic mappings: ecology.{operation} → science.{method} on airSpring primal
# Both science.* and ecology.* namespaces are accepted by the primal.

# ET₀ computation — 7 validated methods + ensemble
"ecology.et0_fao56" = { provider = "airspring", method = "science.et0_fao56" }
"ecology.et0_hargreaves" = { provider = "airspring", method = "science.et0_hargreaves" }
"ecology.et0_priestley_taylor" = { provider = "airspring", method = "science.et0_priestley_taylor" }
"ecology.et0_makkink" = { provider = "airspring", method = "science.et0_makkink" }
"ecology.et0_turc" = { provider = "airspring", method = "science.et0_turc" }
"ecology.et0_hamon" = { provider = "airspring", method = "science.et0_hamon" }
"ecology.et0_blaney_criddle" = { provider = "airspring", method = "science.et0_blaney_criddle" }

# Soil moisture and hydraulics
"ecology.water_balance" = { provider = "airspring", method = "science.water_balance" }
"ecology.richards_1d" = { provider = "airspring", method = "science.richards_1d" }
"ecology.pedotransfer" = { provider = "airspring", method = "science.pedotransfer_saxton_rawls" }
"ecology.soil_moisture_topp" = { provider = "airspring", method = "science.soil_moisture_topp" }
"ecology.scs_cn_runoff" = { provider = "airspring", method = "science.scs_cn_runoff" }
"ecology.green_ampt" = { provider = "airspring", method = "science.green_ampt_infiltration" }

# Crop science
"ecology.yield_response" = { provider = "airspring", method = "science.yield_response" }
"ecology.gdd" = { provider = "airspring", method = "science.gdd" }
"ecology.dual_kc" = { provider = "airspring", method = "science.dual_kc" }

# IoT and scheduling
"ecology.sensor_calibrate" = { provider = "airspring", method = "science.sensor_calibration" }

# Biodiversity
"ecology.shannon_diversity" = { provider = "airspring", method = "science.shannon_diversity" }
"ecology.bray_curtis" = { provider = "airspring", method = "science.bray_curtis" }

# Geophysics coupling
"ecology.anderson_coupling" = { provider = "airspring", method = "science.anderson_coupling" }

# Monthly ET
"ecology.thornthwaite" = { provider = "airspring", method = "science.thornthwaite" }

# Data acquisition
"ecology.fetch_weather" = { provider = "nestgate", method = "storage.retrieve", fallback_provider = "airspring" }

# Full pipeline orchestration
"ecology.full_pipeline" = { provider = "airspring", method = "ecology.full_pipeline" }
```

---

## Socket Registration

airSpring binds at the standard biomeOS socket path:

```
$XDG_RUNTIME_DIR/biomeos/airspring-${FAMILY_ID}.sock
```

JSON-RPC 2.0 over Unix socket. All methods accept and return JSON.

---

## JSON-RPC Method Signatures

### `eco.daily_et0`

FAO-56 Penman-Monteith reference ET₀.

```json
{
  "jsonrpc": "2.0",
  "method": "eco.daily_et0",
  "params": {
    "tmin": 12.3, "tmax": 21.5, "tmean": 16.9,
    "solar_radiation": 22.07, "wind_speed_2m": 2.078,
    "actual_vapour_pressure": 1.409, "elevation_m": 100.0,
    "latitude_deg": 50.8, "day_of_year": 187
  },
  "id": 1
}
```

Response: `{ "et0": 3.88, "rn": 13.28, "delta": 0.122, "gamma": 0.067, ... }`

### `eco.et0_multi_method`

Compute ET₀ using multiple methods on the same weather data.

```json
{
  "jsonrpc": "2.0",
  "method": "eco.et0_multi_method",
  "params": {
    "weather": { "tmin": 12.3, "tmax": 21.5, "rs_mj": 22.07, "wind_speed_2m": 2.078, "rh_pct": 66.5, "elevation_m": 100.0, "latitude_deg": 50.8, "day_of_year": 187 },
    "methods": ["pm", "pt", "hargreaves", "makkink", "turc", "hamon"]
  },
  "id": 2
}
```

Response: `{ "pm": 3.88, "pt": 3.24, "hargreaves": 4.12, "makkink": 2.44, "turc": 3.03, "hamon": 3.99 }`

### `eco.water_balance_season`

FAO-56 Chapter 8 seasonal water balance simulation.

```json
{
  "jsonrpc": "2.0",
  "method": "eco.water_balance_season",
  "params": {
    "et0_series": [3.88, 4.12, 3.95, ...],
    "precip_series": [0.0, 5.2, 0.0, ...],
    "crop": "corn",
    "soil_texture": "loam",
    "strategy": "mad_50"
  },
  "id": 3
}
```

Response: `{ "total_et": 542.0, "total_irrigation": 180.0, "yield_ratio": 0.96, "mass_balance_error": 0.0 }`

---

## Compute Routing (Internal)

airSpring handles GPU dispatch internally via metalForge:

| Workload | CPU Path | GPU Path (ToadStool) | NPU Path (AKD1000) |
|----------|----------|---------------------|---------------------|
| ET₀ batch | `eco::evapotranspiration` | `BatchedEt0` | — |
| Water balance | `eco::water_balance` | `BatchedWaterBalance` | — |
| Richards PDE | `eco::richards` | `BatchedRichards` | — |
| Yield response | `eco::yield_response` | `BatchedElementwise` | — |
| Crop stress | — | — | `npu::CropStressClassifier` |
| Irrigation | — | — | `npu::IrrigationDecision` |

The caller (biomeOS graph or direct) doesn't choose hardware —
airSpring's metalForge dispatch selects GPU > NPU > CPU based on capability.

---

## Cross-Primal Interactions

### airSpring → wetSpring (via biomeOS)

```
ecology.water_balance(θ_series)
  → capability.call("science.diversity", { soil_moisture: θ_series })
  → wetSpring computes microbial diversity under varying moisture
```

### airSpring → groundSpring (via biomeOS)

```
ecology.et0_sensitivity()
  → capability.call("measurement.error_propagation", { et0_inputs: {...}, uncertainties: {...} })
  → groundSpring computes Monte Carlo uncertainty bands
```

### wetSpring → airSpring (via biomeOS)

```
science.soil_microbiome_coupling()
  → capability.call("ecology.water_balance", { et0: [...], precip: [...] })
  → airSpring provides θ(t) time series for microbiome modeling
```

---

## Deployment

### Graph

```bash
biomeos deploy graphs/airspring_eco_pipeline.toml \
  --env FAMILY_ID=$(biomeos family-id) \
  --env NODE_ID=eastgate \
  --env LATITUDE=42.77 \
  --env LONGITUDE=-84.47 \
  --env CROP=corn \
  --env START_DATE=2023-05-01 \
  --env END_DATE=2023-09-30
```

### Direct capability.call

```bash
echo '{"jsonrpc":"2.0","method":"capability.call","params":{"capability":"ecology","operation":"et0_pm","args":{"tmin":12.3,"tmax":21.5,"solar_radiation":22.07,"wind_speed_2m":2.078,"actual_vapour_pressure":1.409,"elevation_m":100.0,"latitude_deg":50.8,"day_of_year":187}},"id":1}' | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/biomeos/neural-api-$(biomeos family-id).sock
```
