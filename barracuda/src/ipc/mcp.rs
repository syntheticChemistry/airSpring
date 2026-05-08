// SPDX-License-Identifier: AGPL-3.0-or-later
//! MCP (Model Context Protocol) tool definitions for Squirrel AI integration.
//!
//! Typed tool schemas that allow Squirrel (or any MCP-compliant AI client)
//! to discover airSpring's ecology capabilities. Each tool maps to an
//! existing JSON-RPC method — no new functionality, just structured
//! discoverability.
//!
//! # Discovery
//!
//! When Squirrel queries `tools/list`, airSpring returns these definitions.
//! Squirrel then calls `tools/call` with the tool name and arguments,
//! which the server dispatches to the existing JSON-RPC handler.

use serde_json::{Value, json};

/// A typed MCP tool definition per the MCP specification.
pub struct McpTool {
    /// Tool name (e.g. `"airspring_et0"`).
    pub name: &'static str,
    /// Human-readable description for the AI model.
    pub description: &'static str,
    /// JSON Schema for the input parameters.
    pub input_schema: fn() -> Value,
}

/// All MCP tools exposed by airSpring.
pub const TOOLS: &[McpTool] = &[
    McpTool {
        name: "airspring_et0",
        description: "Compute reference evapotranspiration (ET₀) using the FAO-56 \
                      Penman-Monteith equation. Requires temperature, humidity, wind speed, \
                      and solar radiation. Returns ET₀ in mm/day plus intermediate values \
                      (slope, psychrometric constant, net radiation).",
        input_schema: et0_schema,
    },
    McpTool {
        name: "airspring_hargreaves",
        description: "Compute ET₀ using the Hargreaves-Samani temperature-based method. \
                      Only requires min/max temperature and extraterrestrial radiation. \
                      Returns ET₀ in mm/day — suitable when full weather data is unavailable.",
        input_schema: hargreaves_schema,
    },
    McpTool {
        name: "airspring_water_balance",
        description: "Step a daily FAO-56 water balance forward one day. Tracks root-zone \
                      depletion, stress coefficient, and actual ET for irrigation scheduling.",
        input_schema: water_balance_schema,
    },
    McpTool {
        name: "airspring_soil_moisture",
        description: "Convert soil dielectric permittivity to volumetric water content \
                      using the Topp equation (Topp et al. 1980). Also provides inverse \
                      conversion and USDA soil texture classification with hydraulic properties.",
        input_schema: soil_moisture_schema,
    },
    McpTool {
        name: "airspring_dual_kc",
        description: "Compute dual crop coefficient (Kcb + Ke) partitioning per FAO-56 Ch 7. \
                      Separates transpiration from evaporation for cover crops, mulch, and \
                      no-till systems. Returns daily Kcb, Ke, ETc, and soil water depletion.",
        input_schema: dual_kc_schema,
    },
    McpTool {
        name: "airspring_richards",
        description: "Solve the 1D Richards equation for unsaturated soil water flow using \
                      implicit Euler with Picard iteration. Returns vertical moisture \
                      profile θ(z,t) for infiltration, drainage, and redistribution.",
        input_schema: richards_schema,
    },
    McpTool {
        name: "airspring_yield_response",
        description: "Estimate crop yield response to water stress using the Stewart (1977) \
                      model. Computes relative yield from ET deficit and crop-specific \
                      yield response factor Ky.",
        input_schema: yield_schema,
    },
    McpTool {
        name: "airspring_spi_drought",
        description: "Compute the Standardized Precipitation Index (SPI) for drought \
                      monitoring at multiple time scales (1, 3, 6, 12 months). Uses gamma \
                      distribution MLE and normal quantile transformation per WMO guidelines.",
        input_schema: spi_schema,
    },
    McpTool {
        name: "airspring_diversity",
        description: "Compute ecological diversity indices (Shannon H', Simpson D, Pielou J, \
                      Bray-Curtis dissimilarity) for community abundance vectors. \
                      Used for soil microbiome analysis and Anderson coupling.",
        input_schema: diversity_schema,
    },
    McpTool {
        name: "airspring_pedotransfer",
        description: "Estimate soil hydraulic properties from texture using the Saxton & Rawls \
                      (2006) pedotransfer functions. Returns field capacity, wilting point, \
                      saturated conductivity, and van Genuchten parameters.",
        input_schema: pedotransfer_schema,
    },
];

fn et0_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tmin": { "type": "number", "description": "Minimum daily temperature (°C)" },
            "tmax": { "type": "number", "description": "Maximum daily temperature (°C)" },
            "rh_min": { "type": "number", "description": "Minimum relative humidity (%)" },
            "rh_max": { "type": "number", "description": "Maximum relative humidity (%)" },
            "wind_speed_2m": { "type": "number", "description": "Wind speed at 2m height (m/s)" },
            "solar_radiation": { "type": "number", "description": "Solar radiation (MJ/m²/day)" },
            "elevation": { "type": "number", "description": "Station elevation (m)" },
            "latitude": { "type": "number", "description": "Station latitude (degrees)" },
            "day_of_year": { "type": "integer", "description": "Julian day of year (1-366)" }
        },
        "required": ["tmin", "tmax", "wind_speed_2m", "solar_radiation", "elevation", "latitude", "day_of_year"]
    })
}

fn hargreaves_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tmin": { "type": "number", "description": "Minimum daily temperature (°C)" },
            "tmax": { "type": "number", "description": "Maximum daily temperature (°C)" },
            "ra": { "type": "number", "description": "Extraterrestrial radiation (MJ/m²/day)" }
        },
        "required": ["tmin", "tmax", "ra"]
    })
}

fn water_balance_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "et0": { "type": "number", "description": "Reference ET₀ (mm/day)" },
            "kc": { "type": "number", "description": "Crop coefficient" },
            "precipitation": { "type": "number", "description": "Precipitation (mm/day)" },
            "irrigation": { "type": "number", "description": "Irrigation applied (mm/day)", "default": 0.0 },
            "depletion": { "type": "number", "description": "Current root zone depletion (mm)" },
            "raw": { "type": "number", "description": "Readily available water (mm)" },
            "taw": { "type": "number", "description": "Total available water (mm)" }
        },
        "required": ["et0", "kc", "precipitation", "depletion", "raw", "taw"]
    })
}

fn soil_moisture_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "permittivity": {
                "type": "number",
                "description": "Dielectric permittivity (unitless, typically 1-80)"
            },
            "soil_texture": {
                "type": "string",
                "enum": ["sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt",
                         "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay",
                         "silty_clay", "clay"],
                "description": "USDA soil texture class (optional, for hydraulic properties)"
            }
        },
        "required": ["permittivity"]
    })
}

fn dual_kc_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "et0_daily": {
                "type": "array", "items": { "type": "number" },
                "description": "Daily ET₀ series (mm/day)"
            },
            "precip_daily": {
                "type": "array", "items": { "type": "number" },
                "description": "Daily precipitation series (mm/day)"
            },
            "kcb": { "type": "number", "description": "Basal crop coefficient" },
            "kc_max": { "type": "number", "description": "Maximum crop coefficient", "default": 1.2 },
            "soil": { "type": "string", "description": "Soil texture name" }
        },
        "required": ["et0_daily", "precip_daily", "kcb"]
    })
}

fn richards_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "soil_texture": { "type": "string", "description": "USDA texture for VG parameters" },
            "n_nodes": { "type": "integer", "description": "Spatial discretization nodes", "default": 50 },
            "depth_m": { "type": "number", "description": "Soil column depth (m)", "default": 1.0 },
            "duration_hours": { "type": "number", "description": "Simulation duration (hours)" },
            "dt_seconds": { "type": "number", "description": "Time step (seconds)", "default": 60.0 },
            "top_flux": { "type": "number", "description": "Top boundary flux (m/s, negative=infiltration)" }
        },
        "required": ["soil_texture", "duration_hours", "top_flux"]
    })
}

fn yield_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "ky": { "type": "number", "description": "Yield response factor (FAO-56 Table 24)" },
            "et_actual": { "type": "number", "description": "Actual evapotranspiration (mm)" },
            "et_max": { "type": "number", "description": "Maximum evapotranspiration (mm)" }
        },
        "required": ["ky", "et_actual", "et_max"]
    })
}

fn spi_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "monthly_precip": {
                "type": "array", "items": { "type": "number" },
                "description": "Monthly precipitation series (mm)"
            },
            "scale": { "type": "integer", "description": "SPI time scale (1, 3, 6, or 12 months)", "default": 3 }
        },
        "required": ["monthly_precip"]
    })
}

fn diversity_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "abundances": {
                "type": "array", "items": { "type": "number" },
                "description": "Abundance vector (one entry per OTU/ASV)"
            }
        },
        "required": ["abundances"]
    })
}

fn pedotransfer_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sand_pct": { "type": "number", "description": "Sand content (%)" },
            "clay_pct": { "type": "number", "description": "Clay content (%)" },
            "om_pct": { "type": "number", "description": "Organic matter content (%)", "default": 2.5 }
        },
        "required": ["sand_pct", "clay_pct"]
    })
}

/// Build the `tools/list` response payload per MCP specification.
#[must_use]
pub fn list_tools() -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": (t.input_schema)(),
            })
        })
        .collect();

    json!({ "tools": tools })
}

/// Map an MCP tool name to the corresponding JSON-RPC method.
#[must_use]
pub fn tool_to_method(tool_name: &str) -> Option<&'static str> {
    use crate::methods as m;
    match tool_name {
        "airspring_et0" => Some(m::ET0_FAO56),
        "airspring_hargreaves" => Some(m::ET0_HARGREAVES),
        "airspring_water_balance" => Some(m::WATER_BALANCE),
        "airspring_soil_moisture" => Some(m::SOIL_MOISTURE_TOPP),
        "airspring_dual_kc" => Some(m::DUAL_KC),
        "airspring_richards" => Some(m::RICHARDS_1D),
        "airspring_yield_response" => Some(m::YIELD_RESPONSE),
        "airspring_spi_drought" => Some(m::SPI_DROUGHT_INDEX),
        "airspring_diversity" => Some(m::SHANNON_DIVERSITY),
        "airspring_pedotransfer" => Some(m::PEDOTRANSFER),
        _ => None,
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: assertions use unwrap")]
mod tests {
    use super::*;

    #[test]
    fn list_tools_returns_all() {
        let response = list_tools();
        let tools = response["tools"].as_array().unwrap();
        assert_eq!(tools.len(), TOOLS.len());
    }

    #[test]
    fn all_tools_have_input_schema() {
        for tool in TOOLS {
            let schema = (tool.input_schema)();
            assert_eq!(
                schema["type"], "object",
                "{} schema must be object",
                tool.name
            );
            assert!(
                schema.get("properties").is_some(),
                "{} schema must have properties",
                tool.name
            );
        }
    }

    #[test]
    fn tool_names_are_prefixed() {
        for tool in TOOLS {
            assert!(
                tool.name.starts_with("airspring_"),
                "MCP tool '{}' must be prefixed with airspring_",
                tool.name
            );
        }
    }

    #[test]
    fn all_tools_have_method_mapping() {
        for tool in TOOLS {
            assert!(
                tool_to_method(tool.name).is_some(),
                "MCP tool '{}' has no JSON-RPC method mapping",
                tool.name
            );
        }
    }

    #[test]
    fn method_mappings_are_registered_capabilities() {
        for tool in TOOLS {
            let method = tool_to_method(tool.name).unwrap();
            assert!(
                crate::niche::CAPABILITIES.contains(&method),
                "tool '{}' maps to '{}' which is not a registered capability",
                tool.name,
                method
            );
        }
    }

    #[test]
    fn unknown_tool_returns_none() {
        assert!(tool_to_method("unknown_tool").is_none());
    }
}
