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

use std::path::{Path, PathBuf};

/// Resolve the biomeOS socket directory from environment.
///
/// Priority: `BIOMEOS_SOCKET_DIR` > `XDG_RUNTIME_DIR/biomeos` > `/run/user/{uid}/biomeos` > `temp_dir/biomeos`.
#[must_use]
pub fn resolve_socket_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg).join("biomeos");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            let uid = meta.uid();
            let dir = PathBuf::from(format!("/run/user/{uid}/biomeos"));
            if dir.parent().is_some_and(Path::exists) {
                return dir;
            }
        }
    }
    std::env::temp_dir().join("biomeos")
}

/// Resolve the primal family ID from environment.
///
/// Priority: `FAMILY_ID` > `BIOMEOS_FAMILY_ID` > `"default"`.
#[must_use]
pub fn get_family_id() -> String {
    if let Ok(id) = std::env::var("FAMILY_ID") {
        return id;
    }
    if let Ok(id) = std::env::var("BIOMEOS_FAMILY_ID") {
        return id;
    }
    "default".to_string()
}

/// Resolve the socket path for a specific primal + family.
#[must_use]
pub fn resolve_socket_path(primal_name: &str, family_id: &str) -> PathBuf {
    resolve_socket_dir().join(format!("{primal_name}-{family_id}.sock"))
}

/// Discover a primal's socket by scanning the socket directory.
///
/// Tries `{name}-{family}.sock` first, then `{name}.sock`, then any
/// file starting with `{name}` and ending with `.sock`.
#[must_use]
pub fn discover_primal_socket(primal_name: &str) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();
    let family_id = get_family_id();

    let with_family = socket_dir.join(format!("{primal_name}-{family_id}.sock"));
    if with_family.exists() {
        return Some(with_family);
    }

    let without_family = socket_dir.join(format!("{primal_name}.sock"));
    if without_family.exists() {
        return Some(without_family);
    }

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(primal_name) && name_str.ends_with(".sock") {
                return Some(entry.path());
            }
        }
    }

    None
}

/// Find a socket by prefix (e.g., `"airspring"` finds `airspring-*.sock`).
#[must_use]
pub fn find_socket(prefix: &str) -> Option<PathBuf> {
    let socket_dir = resolve_socket_dir();
    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(prefix) && s.ends_with(".sock") {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Resolve the fallback registration primal from environment.
///
/// Returns the primal name from `BIOMEOS_FALLBACK_PRIMAL` or `None` if unset.
/// This replaces hardcoded fallback primal names.
#[must_use]
pub fn fallback_registration_primal() -> Option<String> {
    std::env::var("BIOMEOS_FALLBACK_PRIMAL").ok()
}

/// List all discovered primals in the socket directory.
#[must_use]
pub fn discover_all_primals() -> Vec<String> {
    let socket_dir = resolve_socket_dir();
    let mut primals = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&socket_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if Path::new(&name_str)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("sock"))
            {
                let primal_name = name_str
                    .split('-')
                    .next()
                    .unwrap_or(&name_str)
                    .trim_end_matches(".sock")
                    .to_string();

                if !primals.contains(&primal_name) {
                    primals.push(primal_name);
                }
            }
        }
    }

    primals.sort();
    primals
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn resolve_socket_dir_returns_path() {
        let dir = resolve_socket_dir();
        assert!(dir.to_string_lossy().contains("biomeos"));
    }

    #[test]
    fn get_family_id_returns_nonempty() {
        let id = get_family_id();
        assert!(!id.is_empty());
    }

    #[test]
    fn resolve_socket_path_includes_primal() {
        let path = resolve_socket_path("airspring", "test-family");
        let s = path.to_string_lossy();
        assert!(s.contains("airspring"));
        assert!(s.contains("test-family"));
        assert!(s.ends_with(".sock"));
    }

    #[test]
    fn discover_all_primals_returns_vec() {
        let primals = discover_all_primals();
        assert!(primals.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn fallback_registration_primal_reads_env() {
        let _val = fallback_registration_primal();
    }

    #[test]
    #[serial]
    fn resolve_socket_dir_uses_biomeos_socket_dir_when_set() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_test_{}", std::process::id()));
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let result = resolve_socket_dir();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert_eq!(result, unique_dir);
    }

    #[test]
    #[serial]
    fn resolve_socket_dir_uses_xdg_runtime_dir_when_set() {
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        let xdg_dir = std::env::temp_dir().join(format!("xdg_test_{}", std::process::id()));
        std::env::set_var("XDG_RUNTIME_DIR", xdg_dir.as_os_str());
        let result = resolve_socket_dir();
        std::env::remove_var("XDG_RUNTIME_DIR");
        assert_eq!(result, xdg_dir.join("biomeos"));
    }

    #[test]
    #[serial]
    fn get_family_id_uses_family_id_when_set() {
        std::env::set_var("FAMILY_ID", "custom-family-123");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        let id = get_family_id();
        std::env::remove_var("FAMILY_ID");
        assert_eq!(id, "custom-family-123");
    }

    #[test]
    #[serial]
    fn get_family_id_uses_biomeos_family_id_when_only_it_set() {
        std::env::remove_var("FAMILY_ID");
        std::env::set_var("BIOMEOS_FAMILY_ID", "biomeos-family-456");
        let id = get_family_id();
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        assert_eq!(id, "biomeos-family-456");
    }

    #[test]
    #[serial]
    fn get_family_id_returns_default_when_neither_set() {
        std::env::remove_var("FAMILY_ID");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        let id = get_family_id();
        assert_eq!(id, "default");
    }

    #[test]
    #[serial]
    fn resolve_socket_path_format_verification() {
        std::env::set_var("BIOMEOS_SOCKET_DIR", "/tmp/sockdir");
        let path = resolve_socket_path("primal1", "fam1");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        let s = path.to_string_lossy();
        assert!(s.contains("primal1"));
        assert!(s.contains("fam1"));
        assert!(s.ends_with(".sock"));
        assert!(s.contains("primal1-fam1.sock") || s.ends_with("primal1-fam1.sock"));
    }

    #[test]
    #[serial]
    fn discover_primal_socket_none_when_no_matching_socket() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_empty_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let result = discover_primal_socket("nonexistent_primal_xyz123");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn find_socket_none_when_no_matching_socket() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_empty2_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let result = find_socket("nonexistent_prefix_xyz789");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn fallback_registration_primal_some_when_env_set() {
        std::env::set_var("BIOMEOS_FALLBACK_PRIMAL", "fallback-primal-name");
        let result = fallback_registration_primal();
        std::env::remove_var("BIOMEOS_FALLBACK_PRIMAL");
        assert_eq!(result.as_deref(), Some("fallback-primal-name"));
    }

    #[test]
    #[serial]
    fn fallback_registration_primal_none_when_env_unset() {
        std::env::remove_var("BIOMEOS_FALLBACK_PRIMAL");
        let result = fallback_registration_primal();
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn discover_all_primals_sorted_and_deduplicated() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_primals_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert!(primals.windows(2).all(|w| w[0] <= w[1]));
        let unique_count = primals
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(
            primals.len(),
            unique_count,
            "primals should be deduplicated"
        );
    }

    #[test]
    #[serial]
    fn discover_primal_socket_finds_socket_with_family() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_discover_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        let socket_path = unique_dir.join("testprimal-myfamily.sock");
        std::fs::File::create(&socket_path).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        std::env::set_var("FAMILY_ID", "myfamily");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        let result = discover_primal_socket("testprimal");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::env::remove_var("FAMILY_ID");
        std::fs::remove_file(&socket_path).ok();
        assert_eq!(result, Some(socket_path));
    }

    #[test]
    #[serial]
    fn find_socket_finds_socket_by_prefix() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_find_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        let socket_path = unique_dir.join("airspring-default.sock");
        std::fs::File::create(&socket_path).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let result = find_socket("airspring");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(&socket_path).ok();
        assert_eq!(result, Some(socket_path));
    }

    #[test]
    #[serial]
    fn resolve_socket_path_joins_correctly() {
        std::env::set_var("BIOMEOS_SOCKET_DIR", "/tmp/sockets");
        let path = resolve_socket_path("primal", "fam");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        let s = path.to_string_lossy();
        assert!(s.ends_with("primal-fam.sock"));
    }

    #[test]
    #[serial]
    fn discover_primal_socket_finds_socket_without_family() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_nofam_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        let socket_path = unique_dir.join("bareprimal.sock");
        std::fs::File::create(&socket_path).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        std::env::set_var("FAMILY_ID", "otherfamily");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        let result = discover_primal_socket("bareprimal");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::env::remove_var("FAMILY_ID");
        std::fs::remove_file(&socket_path).ok();
        assert_eq!(result, Some(socket_path));
    }

    #[test]
    #[serial]
    fn discover_primal_socket_finds_socket_via_prefix_scan() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_prefix_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        let socket_path = unique_dir.join("prefixprimal-otherfamily.sock");
        std::fs::File::create(&socket_path).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        std::env::set_var("FAMILY_ID", "wrongfamily");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        let result = discover_primal_socket("prefixprimal");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::env::remove_var("FAMILY_ID");
        std::fs::remove_file(&socket_path).ok();
        assert_eq!(result, Some(socket_path));
    }

    #[test]
    #[serial]
    fn discover_primal_socket_none_when_socket_dir_does_not_exist() {
        let nonexistent =
            std::env::temp_dir().join(format!("biomeos_nonexistent_{}", std::process::id()));
        std::fs::remove_dir_all(&nonexistent).ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", nonexistent.as_os_str());
        let result = discover_primal_socket("anyprimal");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn find_socket_none_when_socket_dir_does_not_exist() {
        let nonexistent =
            std::env::temp_dir().join(format!("biomeos_nonexistent2_{}", std::process::id()));
        std::fs::remove_dir_all(&nonexistent).ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", nonexistent.as_os_str());
        let result = find_socket("anyprefix");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert_eq!(result, None);
    }

    #[test]
    #[serial]
    fn discover_all_primals_empty_when_socket_dir_does_not_exist() {
        let nonexistent =
            std::env::temp_dir().join(format!("biomeos_nonexistent3_{}", std::process::id()));
        std::fs::remove_dir_all(&nonexistent).ok();
        std::env::set_var("BIOMEOS_SOCKET_DIR", nonexistent.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        assert!(primals.is_empty());
    }

    #[test]
    #[serial]
    fn discover_all_primals_extracts_names_from_socket_files() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_extract_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::fs::File::create(unique_dir.join("alpha-default.sock")).unwrap();
        std::fs::File::create(unique_dir.join("beta-other.sock")).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(unique_dir.join("alpha-default.sock")).ok();
        std::fs::remove_file(unique_dir.join("beta-other.sock")).ok();
        assert!(primals.contains(&"alpha".to_string()));
        assert!(primals.contains(&"beta".to_string()));
        assert_eq!(primals.len(), 2);
    }

    #[test]
    #[serial]
    fn discover_all_primals_deduplicates_same_primal_multiple_sockets() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_dedup_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::fs::File::create(unique_dir.join("airspring-a.sock")).unwrap();
        std::fs::File::create(unique_dir.join("airspring-b.sock")).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(unique_dir.join("airspring-a.sock")).ok();
        std::fs::remove_file(unique_dir.join("airspring-b.sock")).ok();
        assert_eq!(primals.iter().filter(|p| *p == "airspring").count(), 1);
    }

    #[test]
    #[serial]
    fn discover_all_primals_accepts_uppercase_sock_extension() {
        let unique_dir = std::env::temp_dir().join(format!("biomeos_upper_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::fs::File::create(unique_dir.join("uppercase.SOCK")).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(unique_dir.join("uppercase.SOCK")).ok();
        assert_eq!(primals.len(), 1);
        assert!(
            primals[0].starts_with("uppercase"),
            "primal name should derive from filename: {primals:?}"
        );
    }

    #[test]
    #[serial]
    fn discover_all_primals_ignores_non_sock_files() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_ignore_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::fs::File::create(unique_dir.join("real.sock")).unwrap();
        std::fs::File::create(unique_dir.join("notsock.txt")).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(unique_dir.join("real.sock")).ok();
        std::fs::remove_file(unique_dir.join("notsock.txt")).ok();
        assert_eq!(primals.len(), 1);
        assert_eq!(primals[0], "real");
    }

    #[test]
    #[serial]
    fn discover_all_primals_handles_hyphenated_primal_name() {
        let unique_dir =
            std::env::temp_dir().join(format!("biomeos_hyphen_{}", std::process::id()));
        std::fs::create_dir_all(&unique_dir).unwrap();
        std::fs::File::create(unique_dir.join("my-primal-default.sock")).unwrap();
        std::env::set_var("BIOMEOS_SOCKET_DIR", unique_dir.as_os_str());
        let primals = discover_all_primals();
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
        std::fs::remove_file(unique_dir.join("my-primal-default.sock")).ok();
        assert!(primals.contains(&"my".to_string()));
    }
}
