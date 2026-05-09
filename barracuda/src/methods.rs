// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical method name constants for airSpring capabilities.
//!
//! Single source of truth for all `science.*`, `ecology.*`, `provenance.*`,
//! and infrastructure method strings used across the codebase. Every module
//! that references a capability method name should import from here rather
//! than using inline string literals.
//!
//! Synchronized with [`capability_registry.toml`] via the
//! `capabilities_match_registry` integration test.

// ── Evapotranspiration (7 methods) ──────────────────────────────────

/// FAO-56 Penman-Monteith reference ET₀.
pub const ET0_FAO56: &str = "science.et0_fao56";
/// Hargreaves-Samani temperature-based ET₀.
pub const ET0_HARGREAVES: &str = "science.et0_hargreaves";
/// Priestley-Taylor equilibrium ET₀.
pub const ET0_PRIESTLEY_TAYLOR: &str = "science.et0_priestley_taylor";
/// Makkink radiation-based ET₀.
pub const ET0_MAKKINK: &str = "science.et0_makkink";
/// Turc temperature + radiation ET₀.
pub const ET0_TURC: &str = "science.et0_turc";
/// Hamon temperature-based PET.
pub const ET0_HAMON: &str = "science.et0_hamon";
/// Blaney-Criddle consumptive use ET₀.
pub const ET0_BLANEY_CRIDDLE: &str = "science.et0_blaney_criddle";

// ── Water balance & yield ───────────────────────────────────────────

/// Single-step field-scale water balance.
pub const WATER_BALANCE: &str = "science.water_balance";
/// Stewart yield-response model (Ky).
pub const YIELD_RESPONSE: &str = "science.yield_response";

// ── Soil physics ────────────────────────────────────────────────────

/// 1-D Richards equation (van Genuchten-Mualem).
pub const RICHARDS_1D: &str = "science.richards_1d";
/// SCS curve-number runoff.
pub const SCS_CN_RUNOFF: &str = "science.scs_cn_runoff";
/// Green-Ampt infiltration model.
pub const GREEN_AMPT: &str = "science.green_ampt_infiltration";
/// Topp equation dielectric → VWC.
pub const SOIL_MOISTURE_TOPP: &str = "science.soil_moisture_topp";
/// Saxton-Rawls pedotransfer functions.
pub const PEDOTRANSFER: &str = "science.pedotransfer_saxton_rawls";

// ── Crop & irrigation ──────────────────────────────────────────────

/// FAO-56 dual crop coefficient (Kcb + Ke).
pub const DUAL_KC: &str = "science.dual_kc";
/// Dong-style TDR/capacitance sensor calibration.
pub const SENSOR_CALIBRATION: &str = "science.sensor_calibration";
/// Growing degree-days accumulation.
pub const GDD: &str = "science.gdd";

// ── Biodiversity ────────────────────────────────────────────────────

/// Shannon diversity index (H').
pub const SHANNON_DIVERSITY: &str = "science.shannon_diversity";
/// Bray-Curtis dissimilarity.
pub const BRAY_CURTIS: &str = "science.bray_curtis";

// ── Geophysics coupling ────────────────────────────────────────────

/// Anderson hydromechanical coupling.
pub const ANDERSON_COUPLING: &str = "science.anderson_coupling";

// ── Monthly ET ─────────────────────────────────────────────────────

/// Thornthwaite monthly PET.
pub const THORNTHWAITE: &str = "science.thornthwaite";

// ── Drought & stochastic ───────────────────────────────────────────

/// Standardized Precipitation Index.
pub const SPI_DROUGHT_INDEX: &str = "science.spi_drought_index";
/// Time series autocorrelation.
pub const AUTOCORRELATION: &str = "science.autocorrelation";
/// Incomplete gamma CDF.
pub const GAMMA_CDF: &str = "science.gamma_cdf";

// ── Cross-spring time series ───────────────────────────────────────

/// Generic time series handler.
pub const TIMESERIES: &str = "science.timeseries";

// ── Ecology aliases ────────────────────────────────────────────────

/// Ecology-domain alias for FAO-56 ET₀.
pub const ECO_ET0_FAO56: &str = "ecology.et0_fao56";
/// Ecology-domain alias for Hargreaves ET₀.
pub const ECO_ET0_HARGREAVES: &str = "ecology.et0_hargreaves";
/// Ecology-domain alias for Priestley-Taylor ET₀.
pub const ECO_ET0_PRIESTLEY_TAYLOR: &str = "ecology.et0_priestley_taylor";
/// Ecology-domain alias for Makkink ET₀.
pub const ECO_ET0_MAKKINK: &str = "ecology.et0_makkink";
/// Ecology-domain alias for Turc ET₀.
pub const ECO_ET0_TURC: &str = "ecology.et0_turc";
/// Ecology-domain alias for Hamon PET.
pub const ECO_ET0_HAMON: &str = "ecology.et0_hamon";
/// Ecology-domain alias for Blaney-Criddle ET₀.
pub const ECO_ET0_BLANEY_CRIDDLE: &str = "ecology.et0_blaney_criddle";
/// Ecology-domain alias for water balance.
pub const ECO_WATER_BALANCE: &str = "ecology.water_balance";
/// Ecology-domain alias for yield response.
pub const ECO_YIELD_RESPONSE: &str = "ecology.yield_response";
/// Ecology-domain full pipeline.
pub const ECO_FULL_PIPELINE: &str = "ecology.full_pipeline";
/// Ecology-domain alias for SPI drought index.
pub const ECO_SPI_DROUGHT_INDEX: &str = "ecology.spi_drought_index";
/// Ecology-domain alias for autocorrelation.
pub const ECO_AUTOCORRELATION: &str = "ecology.autocorrelation";
/// Ecology-domain alias for time series.
pub const ECO_TIMESERIES: &str = "ecology.timeseries";

// ── Provenance trio ────────────────────────────────────────────────

/// Begin a provenance-tracked experiment session.
pub const PROVENANCE_BEGIN: &str = "provenance.begin";
/// Record an experiment step.
pub const PROVENANCE_RECORD: &str = "provenance.record";
/// Complete an experiment (dehydrate → commit → attribute).
pub const PROVENANCE_COMPLETE: &str = "provenance.complete";
/// Query provenance availability.
pub const PROVENANCE_STATUS: &str = "provenance.status";

// ── Cross-primal ───────────────────────────────────────────────────

/// Forward a request to another primal.
pub const PRIMAL_FORWARD: &str = "primal.forward";
/// Discover available primals.
pub const PRIMAL_DISCOVER: &str = "primal.discover";

// ── Health probes ──────────────────────────────────────────────────

/// biomeOS liveness probe.
pub const HEALTH_LIVENESS: &str = "health.liveness";
/// biomeOS readiness probe.
pub const HEALTH_READINESS: &str = "health.readiness";

// ── Infrastructure ─────────────────────────────────────────────────

/// List all registered capabilities.
pub const CAPABILITY_LIST: &str = "capability.list";
/// Cross-spring weather data routing via `NestGate`.
pub const DATA_CROSS_SPRING_WEATHER: &str = "data.cross_spring_weather";
/// `ToadStool` compute offload.
pub const COMPUTE_OFFLOAD: &str = "compute.offload";
/// Weather data via Nest Atomic routing.
pub const DATA_WEATHER: &str = "data.weather";
