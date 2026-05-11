# Primal Proof IPC Mapping — airSpring

**Date**: May 9, 2026
**Status**: 46 capabilities IPC-exposed, UniBin eukaryotic, guideStone L2→L4 target

Maps each airSpring domain computation to its JSON-RPC equivalent for
primal-proof validation. When NUCLEUS primals are deployed, all science
should route through IPC rather than direct `barracuda::` library calls.

## Evapotranspiration (7 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::evapotranspiration::daily_et0()` | `science.et0_fao56` | IPC-exposed |
| `eco::evapotranspiration::hargreaves_et0()` | `science.et0_hargreaves` | IPC-exposed |
| `eco::evapotranspiration::priestley_taylor_et0()` | `science.et0_priestley_taylor` | IPC-exposed |
| `eco::simple_et0::makkink_et0()` | `science.et0_makkink` | IPC-exposed |
| `eco::simple_et0::turc_et0()` | `science.et0_turc` | IPC-exposed |
| `eco::simple_et0::hamon_pet_from_location()` | `science.et0_hamon` | IPC-exposed |
| `eco::simple_et0::blaney_criddle_from_location()` | `science.et0_blaney_criddle` | IPC-exposed |

## Water Balance & Yield (2 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::water_balance::WaterBalance::run()` | `science.water_balance` | IPC-exposed |
| `eco::water_balance::yield_response()` | `science.yield_response` | IPC-exposed |

## Soil Physics (4 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::richards::solve_richards_vg()` | `science.richards_1d` | IPC-exposed |
| `eco::runoff::scs_cn_runoff_standard()` | `science.scs_cn_runoff` | IPC-exposed |
| `eco::infiltration::cumulative_infiltration()` | `science.green_ampt_infiltration` | IPC-exposed |
| `eco::soil_moisture::topp_equation()` | `science.soil_moisture_topp` | IPC-exposed |
| `eco::soil_moisture::saxton_rawls_*()` | `science.pedotransfer_saxton_rawls` | IPC-exposed |

## Crop & Irrigation (3 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::dual_kc::*()` | `science.dual_kc` | IPC-exposed |
| `eco::sensor_calibration::soilwatch10_vwc()` | `science.sensor_calibration` | IPC-exposed |
| `eco::crop::gdd_avg()` | `science.gdd` | IPC-exposed |

## Biodiversity (2 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::diversity::shannon()` | `science.shannon_diversity` | IPC-exposed |
| `eco::diversity::bray_curtis()` | `science.bray_curtis` | IPC-exposed |

## Geophysics & Drought (5 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| `eco::anderson::coupling_chain()` | `science.anderson_coupling` | IPC-exposed |
| `eco::thornthwaite::thornthwaite_monthly_et0()` | `science.thornthwaite` | IPC-exposed |
| SPI computation | `science.spi_drought_index` | IPC-exposed |
| Autocorrelation | `science.autocorrelation` | IPC-exposed |
| Gamma CDF | `science.gamma_cdf` | IPC-exposed |

## Ecology Aliases (7 methods)

| Library Call | JSON-RPC Method | Status |
|-------------|-----------------|--------|
| (alias) | `ecology.et0_fao56` | Routes to `science.et0_fao56` |
| (alias) | `ecology.et0_hargreaves` | Routes to `science.et0_hargreaves` |
| (alias) | `ecology.water_balance` | Routes to `science.water_balance` |
| (alias) | `ecology.yield_response` | Routes to `science.yield_response` |
| (alias) | `ecology.full_pipeline` | Routes to full pipeline |
| (alias) | `ecology.spi_drought_index` | Routes to `science.spi_drought_index` |
| (alias) | `ecology.autocorrelation` | Routes to `science.autocorrelation` |

## Cross-Primal (IPC routing)

| Capability | JSON-RPC Method | Target Primal |
|-----------|-----------------|---------------|
| Provenance begin | `provenance.begin` | rhizoCrypt / loamSpine / sweetGrass |
| Provenance record | `provenance.record` | rhizoCrypt / loamSpine / sweetGrass |
| Provenance complete | `provenance.complete` | rhizoCrypt / loamSpine / sweetGrass |
| Provenance status | `provenance.status` | rhizoCrypt / loamSpine / sweetGrass |
| Time series exchange | `science.timeseries` | Cross-spring exchange |
| Forward to primal | `primal.forward` | Any discovered primal |
| Discover primals | `primal.discover` | biomeOS socket directory |
| Compute offload | `compute.offload` | toadStool |
| Weather data | `data.weather` | NestGate |
| Cross-spring weather | `data.cross_spring_weather` | NestGate |

## Health & Discovery

| Capability | JSON-RPC Method | Purpose |
|-----------|-----------------|---------|
| Liveness | `health.liveness` | biomeOS health check |
| Readiness | `health.readiness` | biomeOS deployment readiness |
| Capability list | `capability.list` | Songbird service discovery |

## Not Yet IPC-Exposed (library-only)

These modules run in-process only (validation binaries, GPU pipelines):

| Module | Reason | Priority |
|--------|--------|----------|
| `gpu::et0` | GPU batch — dispatched via `compute.offload` to toadStool | Low (toadStool handles) |
| `gpu::water_balance` | GPU batch — same | Low |
| `gpu::kriging` | GPU spatial interpolation — same | Low |
| `gpu::richards` | GPU PDE solver — same | Low |
| `gpu::atlas_stream` | Multi-station streaming — same | Low |
| `eco::isotherm` | Isotherm fitting — niche internal | Medium |
| `eco::correction` | Sensor correction — niche internal | Medium |
| `eco::tissue` / `eco::cytokine` | Immunological Anderson — paper-specific | Low |
| `nautilus` | Reservoir computing — niche internal | Medium |

GPU computations are not individually IPC-exposed because they route through
`compute.offload` to toadStool, which delegates to barraCuda. The niche
provides the science parameters; the compute infrastructure is the Node
Atomic's responsibility.

## Discovery Stack

airSpring uses the 5-tier IPC discovery pattern:

1. **Songbird capability registry** — `discover_primal_by_capability()`
2. **Neural API** — `NEURAL_API_SOCKET` / `NEURAL_API_ADDRESS` env vars
3. **UDS convention** — `{socket_dir}/{name}-{family}.sock`
4. **Socket directory scan** — `biomeos::discover_all_primals()`
5. **Env override** — `{PRIMAL}_SOCKET` / `{PRIMAL}_ADDRESS` per-primal vars

## Evolution Path

```
Current (L2):  Direct library calls + IPC for cross-primal
Target  (L3):  CompositionContext for all live calls
Target  (L4+): Live NUCLEUS-backed guidestone certification
```

When `CompositionContext` from primalSpring v0.9.25 is wired (via the
`guidestone` feature flag), all primal calls should route through
`ctx.call("capability", "method", params)` instead of direct
`rpc::resolve_transport()` / `rpc::send_to()`.
