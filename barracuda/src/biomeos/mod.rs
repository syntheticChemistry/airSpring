// SPDX-License-Identifier: AGPL-3.0-or-later
//! biomeOS socket resolution and primal discovery.
//!
//! Centralises the socket directory, family ID, and primal discovery
//! logic that was previously duplicated across the airSpring primal
//! binary and both NUCLEUS validation binaries.
//!
//! All path resolution is environment-driven and capability-based:
//!
//! | Env Var | Purpose |
//! |---------|---------|
//! | `BIOMEOS_SOCKET_DIR` | Override socket directory |
//! | `XDG_RUNTIME_DIR` | Standard XDG fallback |
//! | `BIOMEOS_FAMILY_ID` / `FAMILY_ID` | Primal family multiplexing |
//! | `BIOMEOS_FALLBACK_PRIMAL` | Fallback registration target (instead of hardcoded name) |
//!
//! # Dependency Injection
//!
//! Every public function has a `_with` or `_in` variant that accepts
//! explicit parameters instead of reading the environment.  Tests use
//! these directly — no `set_var` / `remove_var`, no `unsafe`, no
//! `#[serial]`.  The top-level wrappers simply delegate:
//!
//! ```rust,no_run
//! # use std::path::PathBuf;
//! # struct SocketConfig;
//! # impl SocketConfig { fn from_env() -> Self { Self } }
//! # fn resolve_socket_dir_with(_: &SocketConfig) -> PathBuf { PathBuf::new() }
//! pub fn resolve_socket_dir() -> PathBuf {
//!     resolve_socket_dir_with(&SocketConfig::from_env())
//! }
//! ```

use std::path::PathBuf;

mod capabilities;
mod discovery;

pub use capabilities::parse_capabilities;
pub use discovery::{
    discover_all_primals, discover_all_primals_in, discover_inference_primal,
    discover_primal_by_capability, discover_primal_socket, discover_primal_socket_in,
    discover_shader_compiler, discover_visualization_primal, find_socket, find_socket_in,
};

/// Explicit configuration for biomeOS socket resolution.
///
/// Mirrors the environment variables but accepts them as plain values,
/// enabling dependency injection for tests and embedded use.
#[derive(Debug, Clone, Default)]
pub struct SocketConfig {
    /// Overrides `BIOMEOS_SOCKET_DIR`.
    pub socket_dir: Option<PathBuf>,
    /// Overrides `XDG_RUNTIME_DIR`.
    pub xdg_runtime_dir: Option<PathBuf>,
    /// Overrides `FAMILY_ID`.
    pub family_id: Option<String>,
    /// Overrides `BIOMEOS_FAMILY_ID`.
    pub biomeos_family_id: Option<String>,
    /// Overrides `BIOMEOS_FALLBACK_PRIMAL`.
    pub fallback_primal: Option<String>,
}

impl SocketConfig {
    /// Build config by reading the environment (the default path).
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            socket_dir: std::env::var("BIOMEOS_SOCKET_DIR").ok().map(PathBuf::from),
            xdg_runtime_dir: std::env::var("XDG_RUNTIME_DIR").ok().map(PathBuf::from),
            family_id: std::env::var("FAMILY_ID").ok(),
            biomeos_family_id: std::env::var("BIOMEOS_FAMILY_ID").ok(),
            fallback_primal: std::env::var("BIOMEOS_FALLBACK_PRIMAL").ok(),
        }
    }
}

// ── Core implementations (dependency-injected) ───────────────────────

/// Resolve the biomeOS socket directory from explicit config.
///
/// Priority: `socket_dir` > `xdg_runtime_dir/biomeos` > `/run/user/{uid}/biomeos` > `temp_dir/biomeos`.
#[must_use]
pub fn resolve_socket_dir_with(config: &SocketConfig) -> PathBuf {
    if let Some(dir) = &config.socket_dir {
        return dir.clone();
    }
    if let Some(xdg) = &config.xdg_runtime_dir {
        return xdg.join(crate::primal_names::BIOMEOS);
    }
    platform_fallback_socket_dir()
}

/// Resolve the primal family ID from explicit config.
///
/// Priority: `family_id` > `biomeos_family_id` > `"default"`.
#[must_use]
pub fn get_family_id_with(config: &SocketConfig) -> String {
    if let Some(id) = &config.family_id {
        return id.clone();
    }
    if let Some(id) = &config.biomeos_family_id {
        return id.clone();
    }
    "default".to_string()
}

/// Resolve the fallback registration primal from explicit config.
#[must_use]
pub fn fallback_registration_primal_with(config: &SocketConfig) -> Option<String> {
    config.fallback_primal.clone()
}

// ── Public wrappers (read from environment) ──────────────────────────

/// Resolve the biomeOS socket directory from environment.
///
/// Priority: `BIOMEOS_SOCKET_DIR` > `XDG_RUNTIME_DIR/biomeos` > `/run/user/{uid}/biomeos` > `temp_dir/biomeos`.
#[must_use]
pub fn resolve_socket_dir() -> PathBuf {
    resolve_socket_dir_with(&SocketConfig::from_env())
}

/// Resolve the primal family ID from environment.
///
/// Priority: `FAMILY_ID` > `BIOMEOS_FAMILY_ID` > `"default"`.
#[must_use]
pub fn get_family_id() -> String {
    get_family_id_with(&SocketConfig::from_env())
}

/// Resolve the socket path for a specific primal + family.
#[must_use]
pub fn resolve_socket_path(primal_name: &str, family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("{primal_name}-{family_id}.sock"))
}

/// Resolve the fallback registration primal from environment.
///
/// Returns the primal name from `BIOMEOS_FALLBACK_PRIMAL` or `None` if unset.
#[must_use]
pub fn fallback_registration_primal() -> Option<String> {
    fallback_registration_primal_with(&SocketConfig::from_env())
}

// ── Platform fallback ────────────────────────────────────────────────

fn platform_fallback_socket_dir() -> PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(std::path::Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join(crate::primal_names::BIOMEOS)
}

// ── Tests (zero unsafe, zero #[serial]) ──────────────────────────────

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code uses unwrap for clarity")]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn tmp_dir(suffix: &str) -> PathBuf {
        std::env::temp_dir().join(format!("biomeos_test_{suffix}_{}", std::process::id()))
    }

    #[test]
    fn resolve_socket_dir_with_uses_socket_dir_first() {
        let dir = tmp_dir("sockdir");
        let config = SocketConfig {
            socket_dir: Some(dir.clone()),
            xdg_runtime_dir: Some(PathBuf::from("/should/not/use")),
            ..Default::default()
        };
        assert_eq!(resolve_socket_dir_with(&config), dir);
    }

    #[test]
    fn resolve_socket_dir_with_uses_xdg_when_no_socket_dir() {
        let xdg = tmp_dir("xdg");
        let config = SocketConfig {
            xdg_runtime_dir: Some(xdg.clone()),
            ..Default::default()
        };
        assert_eq!(resolve_socket_dir_with(&config), xdg.join("biomeos"));
    }

    #[test]
    fn resolve_socket_dir_with_falls_back_to_platform() {
        let config = SocketConfig::default();
        let dir = resolve_socket_dir_with(&config);
        assert!(dir.to_string_lossy().contains("biomeos"));
    }

    #[test]
    fn get_family_id_with_uses_family_id_first() {
        let config = SocketConfig {
            family_id: Some("custom-123".into()),
            biomeos_family_id: Some("should-not-use".into()),
            ..Default::default()
        };
        assert_eq!(get_family_id_with(&config), "custom-123");
    }

    #[test]
    fn get_family_id_with_uses_biomeos_family_id_when_no_family_id() {
        let config = SocketConfig {
            biomeos_family_id: Some("biomeos-456".into()),
            ..Default::default()
        };
        assert_eq!(get_family_id_with(&config), "biomeos-456");
    }

    #[test]
    fn get_family_id_with_returns_default_when_neither_set() {
        let config = SocketConfig::default();
        assert_eq!(get_family_id_with(&config), "default");
    }

    #[test]
    fn resolve_socket_path_includes_primal_and_family() {
        let path = resolve_socket_path("airspring", "test-family");
        let s = path.to_string_lossy();
        assert!(s.contains("airspring"));
        assert!(s.contains("test-family"));
        assert!(s.ends_with(".sock"));
    }

    #[test]
    fn resolve_socket_path_format() {
        let config = SocketConfig {
            socket_dir: Some(PathBuf::from("/sockets")),
            ..Default::default()
        };
        let dir = resolve_socket_dir_with(&config);
        let path = dir.join("primal1-fam1.sock");
        let s = path.to_string_lossy();
        assert!(s.contains("primal1-fam1.sock"));
    }

    #[test]
    fn discover_primal_socket_in_finds_socket_with_family() {
        let dir = tmp_dir("discover_fam");
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("testprimal-myfamily.sock");
        std::fs::File::create(&sock).unwrap();

        let result = discover_primal_socket_in("testprimal", &dir, "myfamily");
        std::fs::remove_file(&sock).ok();
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn discover_primal_socket_in_finds_socket_without_family() {
        let dir = tmp_dir("discover_nofam");
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("bareprimal.sock");
        std::fs::File::create(&sock).unwrap();

        let result = discover_primal_socket_in("bareprimal", &dir, "otherfamily");
        std::fs::remove_file(&sock).ok();
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn discover_primal_socket_in_finds_via_prefix_scan() {
        let dir = tmp_dir("discover_prefix");
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("prefixprimal-otherfamily.sock");
        std::fs::File::create(&sock).unwrap();

        let result = discover_primal_socket_in("prefixprimal", &dir, "wrongfamily");
        std::fs::remove_file(&sock).ok();
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn discover_primal_socket_in_none_when_no_matching_socket() {
        let dir = tmp_dir("discover_empty");
        std::fs::create_dir_all(&dir).unwrap();

        let result = discover_primal_socket_in("nonexistent_xyz", &dir, "default");
        assert_eq!(result, None);
    }

    #[test]
    fn discover_primal_socket_in_none_when_dir_missing() {
        let dir = tmp_dir("discover_nonexist");
        std::fs::remove_dir_all(&dir).ok();

        let result = discover_primal_socket_in("anyprimal", &dir, "default");
        assert_eq!(result, None);
    }

    #[test]
    fn find_socket_in_finds_by_prefix() {
        let dir = tmp_dir("find_prefix");
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("airspring-default.sock");
        std::fs::File::create(&sock).unwrap();

        let result = find_socket_in("airspring", &dir);
        std::fs::remove_file(&sock).ok();
        assert_eq!(result, Some(sock));
    }

    #[test]
    fn find_socket_in_none_when_no_match() {
        let dir = tmp_dir("find_empty");
        std::fs::create_dir_all(&dir).unwrap();

        let result = find_socket_in("nonexistent_xyz", &dir);
        assert_eq!(result, None);
    }

    #[test]
    fn find_socket_in_none_when_dir_missing() {
        let dir = tmp_dir("find_nonexist");
        std::fs::remove_dir_all(&dir).ok();

        let result = find_socket_in("anyprefix", &dir);
        assert_eq!(result, None);
    }

    #[test]
    fn fallback_registration_primal_with_some() {
        let config = SocketConfig {
            fallback_primal: Some("fallback-name".into()),
            ..Default::default()
        };
        assert_eq!(
            fallback_registration_primal_with(&config).as_deref(),
            Some("fallback-name")
        );
    }

    #[test]
    fn fallback_registration_primal_with_none() {
        let config = SocketConfig::default();
        assert_eq!(fallback_registration_primal_with(&config), None);
    }

    #[test]
    fn discover_all_primals_in_sorted_and_deduplicated() {
        let dir = tmp_dir("all_sorted");
        std::fs::create_dir_all(&dir).unwrap();

        let primals = discover_all_primals_in(&dir);
        assert!(primals.windows(2).all(|w| w[0] <= w[1]));
        let unique_count = primals
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(primals.len(), unique_count);
    }

    #[test]
    fn discover_all_primals_in_empty_when_dir_missing() {
        let dir = tmp_dir("all_nonexist");
        std::fs::remove_dir_all(&dir).ok();

        let primals = discover_all_primals_in(&dir);
        assert!(primals.is_empty());
    }

    #[test]
    fn discover_all_primals_in_extracts_names() {
        let dir = tmp_dir("all_extract");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join("alpha-default.sock")).unwrap();
        std::fs::File::create(dir.join("beta-other.sock")).unwrap();

        let primals = discover_all_primals_in(&dir);
        std::fs::remove_file(dir.join("alpha-default.sock")).ok();
        std::fs::remove_file(dir.join("beta-other.sock")).ok();

        assert!(primals.contains(&"alpha".to_string()));
        assert!(primals.contains(&"beta".to_string()));
        assert_eq!(primals.len(), 2);
    }

    #[test]
    fn discover_all_primals_in_deduplicates() {
        let dir = tmp_dir("all_dedup");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join("airspring-a.sock")).unwrap();
        std::fs::File::create(dir.join("airspring-b.sock")).unwrap();

        let primals = discover_all_primals_in(&dir);
        std::fs::remove_file(dir.join("airspring-a.sock")).ok();
        std::fs::remove_file(dir.join("airspring-b.sock")).ok();

        assert_eq!(primals.iter().filter(|p| *p == "airspring").count(), 1);
    }

    #[test]
    fn discover_all_primals_in_accepts_uppercase_sock() {
        let dir = tmp_dir("all_upper");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join("uppercase.SOCK")).unwrap();

        let primals = discover_all_primals_in(&dir);
        std::fs::remove_file(dir.join("uppercase.SOCK")).ok();

        assert_eq!(primals.len(), 1);
        assert!(
            primals[0].starts_with("uppercase"),
            "primal name should derive from filename: {primals:?}"
        );
    }

    #[test]
    fn discover_all_primals_in_ignores_non_sock_files() {
        let dir = tmp_dir("all_ignore");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join("real.sock")).unwrap();
        std::fs::File::create(dir.join("notsock.txt")).unwrap();

        let primals = discover_all_primals_in(&dir);
        std::fs::remove_file(dir.join("real.sock")).ok();
        std::fs::remove_file(dir.join("notsock.txt")).ok();

        assert_eq!(primals.len(), 1);
        assert_eq!(primals[0], "real");
    }

    #[test]
    fn discover_all_primals_in_handles_hyphenated_name() {
        let dir = tmp_dir("all_hyphen");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::File::create(dir.join("my-primal-default.sock")).unwrap();

        let primals = discover_all_primals_in(&dir);
        std::fs::remove_file(dir.join("my-primal-default.sock")).ok();

        assert!(primals.contains(&"my".to_string()));
    }

    #[test]
    fn env_wrappers_return_valid_results() {
        let dir = resolve_socket_dir();
        assert!(dir.to_string_lossy().contains("biomeos"));

        let id = get_family_id();
        assert!(!id.is_empty());

        let _all = discover_all_primals();
        let _fb = fallback_registration_primal();
    }

    #[test]
    fn parse_capabilities_flat_string_array() {
        let val = serde_json::json!(["health", "compute.dispatch", "data.weather"]);
        let caps = parse_capabilities(&val);
        assert_eq!(caps, vec!["health", "compute.dispatch", "data.weather"]);
    }

    #[test]
    fn parse_capabilities_object_array() {
        let val = serde_json::json!([
            {"name": "health", "version": "1.0"},
            {"capability": "compute.dispatch"}
        ]);
        let caps = parse_capabilities(&val);
        assert_eq!(caps, vec!["health", "compute.dispatch"]);
    }

    #[test]
    fn parse_capabilities_nested_wrapper() {
        let val = serde_json::json!({"capabilities": ["health", "data.weather"]});
        let caps = parse_capabilities(&val);
        assert_eq!(caps, vec!["health", "data.weather"]);
    }

    #[test]
    fn parse_capabilities_double_nested() {
        let val = serde_json::json!({
            "capabilities": {"capabilities": ["health", "compute.dispatch"]}
        });
        let caps = parse_capabilities(&val);
        assert_eq!(caps, vec!["health", "compute.dispatch"]);
    }

    #[test]
    fn parse_capabilities_result_wrapper() {
        let val = serde_json::json!({"result": ["health", "data.weather"]});
        let caps = parse_capabilities(&val);
        assert_eq!(caps, vec!["health", "data.weather"]);
    }

    #[test]
    fn parse_capabilities_empty() {
        let val = serde_json::json!(null);
        assert!(parse_capabilities(&val).is_empty());
        assert!(parse_capabilities(&serde_json::json!(42)).is_empty());
        assert!(parse_capabilities(&serde_json::json!([])).is_empty());
    }
}
