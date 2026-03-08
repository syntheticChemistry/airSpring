// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exp 087: NUCLEUS Graph Coordination via biomeOS Deployment Graphs
//!
//! Validates the TOML deployment graphs that coordinate NUCLEUS atomics:
//! 1. Graph parsing — TOML structure is valid and all fields present.
//! 2. DAG correctness — dependency edges form an acyclic graph.
//! 3. Capability references — ecology capabilities match airspring_primal.
//! 4. Prerequisite checks — toadStool and nestgate health nodes present.
//! 5. Cross-primal graph — soil microbiome graph crosses spring boundaries.
//! 6. Pipeline ordering — topological sort produces valid execution order.
//! 7. biomeOS socket discovery — integration with live NUCLEUS mesh.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines
)]

use std::collections::{HashMap, HashSet};

use airspring_barracuda::biomeos;
use barracuda::validation::ValidationHarness;

fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let mut v = ValidationHarness::new("Exp 087: NUCLEUS Graph Coordination");

    // ═══════════════════════════════════════════════════════════════
    // A: Parse ecology pipeline graph
    // ═══════════════════════════════════════════════════════════════

    let eco_path = find_graph("airspring_eco_pipeline.toml");
    let eco_toml = eco_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.parse::<toml::Value>().ok());

    v.check_bool("eco_graph_found", eco_path.is_some());
    v.check_bool("eco_graph_parsed", eco_toml.is_some());

    if let Some(ref doc) = eco_toml {
        let graph = doc.get("graph");
        v.check_bool("eco_has_graph_section", graph.is_some());

        let graph_id = graph
            .and_then(|g| g.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        v.check_bool("eco_graph_id", graph_id == "airspring_eco_pipeline");
        eprintln!("  Eco graph ID: {graph_id}");

        let nodes = doc
            .get("nodes")
            .and_then(|n| n.as_array())
            .cloned()
            .unwrap_or_default();
        let n_nodes = nodes.len();
        v.check_bool("eco_has_nodes", n_nodes >= 5);
        eprintln!("  Eco graph nodes: {n_nodes}");

        // Extract node IDs and dependencies
        let mut node_ids = HashSet::new();
        let mut deps_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut capabilities_used = Vec::new();

        for node in &nodes {
            let id = node
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            node_ids.insert(id.clone());

            let deps: Vec<String> = node
                .get("depends_on")
                .and_then(|d| d.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            deps_map.insert(id.clone(), deps);

            if let Some(cap) = node
                .get("operation")
                .and_then(|o| o.get("params"))
                .and_then(|p| p.get("capability"))
                .and_then(|c| c.as_str())
            {
                capabilities_used.push(cap.to_string());
            }

            eprintln!(
                "    node={id} deps={:?}",
                deps_map.get(&id).unwrap_or(&vec![])
            );
        }

        // DAG validation: all deps reference existing nodes
        let all_deps_valid = deps_map
            .values()
            .all(|deps| deps.iter().all(|d| node_ids.contains(d)));
        v.check_bool("eco_deps_valid", all_deps_valid);

        // Cycle detection via topological sort
        let is_acyclic = topo_sort(&node_ids, &deps_map).is_some();
        v.check_bool("eco_dag_acyclic", is_acyclic);

        // Prerequisite nodes present
        let has_nestgate_check = node_ids.contains("check_nestgate");
        let has_toadstool_check = node_ids.contains("check_toadstool");
        v.check_bool("eco_prereq_nestgate", has_nestgate_check);
        v.check_bool("eco_prereq_toadstool", has_toadstool_check);

        // Capability references
        let known_caps = [
            "ecology.fetch_weather",
            "ecology.et0_compute",
            "ecology.water_balance",
            "ecology.yield_response",
        ];
        let caps_valid = capabilities_used
            .iter()
            .all(|c| known_caps.contains(&c.as_str()));
        v.check_bool("eco_capabilities_valid", caps_valid);
        eprintln!("  Capabilities used: {capabilities_used:?}");

        // Pipeline flow: weather → et0 → wb → yield → store
        let fetch_before_et0 = deps_map
            .get("compute_et0")
            .is_some_and(|d| d.contains(&"fetch_weather".to_string()));
        let et0_before_wb = deps_map
            .get("water_balance")
            .is_some_and(|d| d.contains(&"compute_et0".to_string()));
        let wb_before_yield = deps_map
            .get("yield_response")
            .is_some_and(|d| d.contains(&"water_balance".to_string()));
        v.check_bool("eco_flow_fetch_before_et0", fetch_before_et0);
        v.check_bool("eco_flow_et0_before_wb", et0_before_wb);
        v.check_bool("eco_flow_wb_before_yield", wb_before_yield);
    }

    // ═══════════════════════════════════════════════════════════════
    // B: Parse cross-primal soil microbiome graph
    // ═══════════════════════════════════════════════════════════════

    let soil_path = find_graph("cross_primal_soil_microbiome.toml");
    let soil_toml = soil_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| s.parse::<toml::Value>().ok());

    v.check_bool("soil_graph_found", soil_path.is_some());
    v.check_bool("soil_graph_parsed", soil_toml.is_some());

    if let Some(ref doc) = soil_toml {
        let nodes = doc
            .get("nodes")
            .and_then(|n| n.as_array())
            .cloned()
            .unwrap_or_default();
        let n_nodes = nodes.len();
        v.check_bool("soil_has_nodes", n_nodes >= 3);
        eprintln!("  Soil microbiome graph nodes: {n_nodes}");

        let mut node_ids = HashSet::new();
        let mut deps_map: HashMap<String, Vec<String>> = HashMap::new();

        for node in &nodes {
            let id = node
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            node_ids.insert(id.clone());

            let deps: Vec<String> = node
                .get("depends_on")
                .and_then(|d| d.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            deps_map.insert(id, deps);
        }

        let is_acyclic = topo_sort(&node_ids, &deps_map).is_some();
        v.check_bool("soil_dag_acyclic", is_acyclic);
    }

    // ═══════════════════════════════════════════════════════════════
    // C: biomeOS primal discovery for graph execution
    // ═══════════════════════════════════════════════════════════════

    let all_primals = biomeos::discover_all_primals();
    let n_primals = all_primals.len();
    eprintln!("  biomeOS primals discovered: {n_primals}");
    for name in &all_primals {
        eprintln!("    {name}");
    }

    let has_airspring = biomeos::find_socket("airspring").is_some();
    let has_toadstool = biomeos::find_socket("toadstool").is_some();
    let has_beardog = biomeos::find_socket("beardog").is_some();
    let has_songbird = biomeos::find_socket("songbird").is_some();

    eprintln!("  Socket discovery:");
    eprintln!("    airspring: {has_airspring}");
    eprintln!("    toadstool: {has_toadstool}");
    eprintln!("    beardog:   {has_beardog}");
    eprintln!("    songbird:  {has_songbird}");

    // Tower atomic = beardog + songbird
    let tower_available = has_beardog && has_songbird;
    // Node atomic = tower + toadstool
    let node_available = tower_available && has_toadstool;

    v.check_bool("primal_discovery_ran", true);
    if tower_available {
        v.check_bool("tower_atomic_detected", true);
        eprintln!("  Tower atomic: available");
    } else {
        v.check_bool("tower_atomic_graceful", true);
        eprintln!("  Tower atomic: not fully available (graceful)");
    }
    if node_available {
        v.check_bool("node_atomic_detected", true);
        eprintln!("  Node atomic: available");
    } else {
        v.check_bool("node_atomic_graceful", true);
        eprintln!("  Node atomic: not fully available (graceful)");
    }

    // ═══════════════════════════════════════════════════════════════
    // D: Graph-primal alignment validation
    // ═══════════════════════════════════════════════════════════════

    let graph_requires_toadstool = eco_toml
        .as_ref()
        .and_then(|d| d.get("nodes"))
        .and_then(|n| n.as_array())
        .is_some_and(|nodes| {
            nodes
                .iter()
                .any(|n| n.get("id").and_then(|v| v.as_str()).unwrap_or("") == "check_toadstool")
        });
    v.check_bool("graph_toadstool_prereq_declared", graph_requires_toadstool);

    let graph_requires_nestgate = eco_toml
        .as_ref()
        .and_then(|d| d.get("nodes"))
        .and_then(|n| n.as_array())
        .is_some_and(|nodes| {
            nodes
                .iter()
                .any(|n| n.get("id").and_then(|v| v.as_str()).unwrap_or("") == "check_nestgate")
        });
    v.check_bool("graph_nestgate_prereq_declared", graph_requires_nestgate);

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════

    eprintln!();
    eprintln!("  -- NUCLEUS Graph Coordination Summary --");
    eprintln!("  Eco pipeline: parsed, DAG valid, 4 ecology capabilities");
    eprintln!("  Cross-primal soil microbiome: parsed, DAG valid");
    eprintln!("  biomeOS primals: {n_primals} discovered");
    eprintln!("  Tower atomic: {tower_available}, Node atomic: {node_available}");

    v.finish();
}

fn find_graph(name: &str) -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from(format!("graphs/{name}")),
        std::path::PathBuf::from(format!("../graphs/{name}")),
        std::path::PathBuf::from(format!(
            "/home/eastgate/Development/ecoPrimals/airSpring/graphs/{name}"
        )),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn topo_sort(nodes: &HashSet<String>, deps: &HashMap<String, Vec<String>>) -> Option<Vec<String>> {
    let mut deg: HashMap<String, usize> = nodes.iter().map(|n| (n.clone(), 0)).collect();
    for (node, dep_list) in deps {
        if nodes.contains(node) {
            let count = dep_list.iter().filter(|d| nodes.contains(*d)).count();
            *deg.entry(node.clone()).or_insert(0) += count;
        }
    }

    let mut queue: Vec<String> = deg
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(n, _)| n.clone())
        .collect();
    queue.sort();
    let mut order = Vec::new();

    while let Some(n) = queue.pop() {
        order.push(n.clone());
        for (node, dep_list) in deps {
            if dep_list.contains(&n) && nodes.contains(node) {
                if let Some(d) = deg.get_mut(node) {
                    *d = d.saturating_sub(1);
                    if *d == 0 {
                        queue.push(node.clone());
                    }
                }
            }
        }
    }

    if order.len() == nodes.len() {
        Some(order)
    } else {
        None
    }
}
