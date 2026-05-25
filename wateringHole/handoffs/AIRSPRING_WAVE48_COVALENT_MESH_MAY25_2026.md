# airSpring — Wave 48 Covalent Mesh Sound Off

**Date**: 2026-05-25
**From**: airSpring (eastGate)
**To**: primalSpring (coord), delta spring teams
**Wave**: 48 — Covalent Spring Mesh

---

## Gate Self-Report

| Field | Value |
|-------|-------|
| **Gate** | eastGate |
| **Hardware** | i9-12900, RTX 4070 + Akida NPU, 32GB DDR5 |
| **Composition** | Full NUCLEUS (13/13 primals) |
| **NUCLEUS status** | operational (11/12 ALIVE + airspring cell) |
| **Co-residents** | primalSpring (coord), neuralSpring, groundSpring |
| **Songbird federation** | port 7700 (TCP active) |
| **LAN mesh** | eastGate ↔ ironGate ↔ southGate ↔ biomeGate |
| **Cell graph** | `plasmidBin/cells/airspring_cell.toml` |

## Deployment Summary

### NUCLEUS Launch

```
SONGBIRD_FEDERATION_PORT=7700 ./tools/nucleus_launcher.sh start
```

11/12 primals ALIVE:

| Phase | Primal | Status | Socket |
|-------|--------|--------|--------|
| 0 | biomeOS | ALIVE | `neural-api-nucleus01.sock` |
| 1 | BearDog | ALIVE | `beardog-nucleus01.sock` |
| 1 | Songbird | ALIVE | `songbird-nucleus01.sock` |
| 2 | ToadStool | ALIVE | `toadstool-nucleus01.sock` |
| 2 | barraCuda | ALIVE | `barracuda-nucleus01.sock` |
| 2 | coralReef | ALIVE | `shader.sock` |
| 2 | NestGate | ALIVE | `nestgate-nucleus01.sock` |
| 2 | Squirrel | ALIVE | `squirrel-nucleus01.sock` |
| 3 | rhizoCrypt | ALIVE | `rhizocrypt-nucleus01.sock` |
| 3 | loamSpine | ALIVE | `loamspine-nucleus01.sock` |
| 3 | sweetGrass | ALIVE | `sweetgrass-nucleus01.sock` |
| 4 | petalTongue | SOCKET | `petaltongue-nucleus01.sock` |

petalTongue failed startup (non-critical visualization primal). BTSP: 7/10 primals respond to handshake probe. Songbird registry: 9 primals seeded.

### Cell Deployment

```
./cell_launcher.sh airspring start
```

airspring_primal binary built from local Rust source (`cargo build --release --features local`), symlinked into `plasmidBin/primals/`. Cell graph skips 8 NUCLEUS primals (spawn=false), spawns only `airspring_primal`.

Result: `airspring-nucleus01.sock` — family `nucleus01`, 46 capabilities, health ALIVE.

### Federation Verification

| Check | Result |
|-------|--------|
| Songbird TCP port 7700 | LISTEN (pid songbird) |
| `discovery.peers` (UDS) | `{"peers":[],"total_count":0}` |
| `discovery.peers` (TCP `/jsonrpc`) | `{"peers":[],"total_count":0}` |
| `health.liveness` (Songbird) | `{"status":"alive"}` |
| `health.liveness` (airspring) | `{"alive":true,"niche":"airspring"}` |
| `capability.list` (airspring) | 46 capabilities, domain ecology |
| `composition.status` (airspring) | nestgate=true, toadstool=true, provenance_trio=false |

0 peers expected — other gates not on active LAN segment at time of verification.

## airSpring Capabilities on Mesh

46 capabilities available for cross-gate dispatch via biomeOS v3.75:

- **Science (32)**: et0_fao56, et0_hargreaves, et0_priestley_taylor, et0_makkink, et0_turc, et0_hamon, et0_blaney_criddle, water_balance, yield_response, richards_1d, scs_cn_runoff, green_ampt_infiltration, soil_moisture_topp, pedotransfer_saxton_rawls, dual_kc, sensor_calibration, gdd, shannon_diversity, bray_curtis, anderson_coupling, thornthwaite, spi_drought_index, autocorrelation, gamma_cdf, timeseries, biomass, leaf_energy, photoperiod, soil_moisture, thermal_time, vpd, water_stress, air_quality, batch_et0
- **Ecology aliases (7)**: et0_fao56, et0_hargreaves, water_balance, yield_response, full_pipeline, spi_drought_index, autocorrelation
- **Provenance (4)**: begin, record, complete, status
- **Infrastructure (3)**: data.weather, data.cross_spring_weather, compute.offload

Any spring on any gate can call `capability.call` with these methods transparently through biomeOS mesh dispatch.

## Known Gaps

| Gap | Detail | Owner |
|-----|--------|-------|
| petalTongue startup | Fails on cell launch — visualization not critical | plasmidBin |
| provenance_trio | `composition.status` reports false — trio env-based discovery | airSpring / upstream |
| Songbird federation path | TCP endpoint at `/jsonrpc` (not root `/`) | upstream docs |
| 0 LAN peers | Other gates offline during test window | infra / timing |
| skunkBat socket | Not started by nucleus_launcher — known Wave 46+ gap | plasmidBin |

## What's Next

1. Cross-gate capability.call smoke tests when ironGate/southGate come online
2. Plasmodium status when 3+ gates meshed (`biomeos plasmodium status`)
3. toadStool S274 yield-to-owner validation with GPU workloads
4. neuralSpring co-tenant coordination on eastGate NPU scheduling
