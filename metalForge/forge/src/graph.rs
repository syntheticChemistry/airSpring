// SPDX-License-Identifier: AGPL-3.0-or-later

//! `biomeOS` graph execution — parse and execute TOML-defined DAGs.
//!
//! `biomeOS` graphs define pipelines as directed acyclic graphs where
//! each node maps to a workload dispatched via `metalForge`. The graph
//! engine:
//!
//! 1. Parses the TOML graph definition
//! 2. Computes topological execution order
//! 3. Maps each node to a substrate via capability-based dispatch
//! 4. Computes transfer paths between consecutive stages
//!
//! # Example Graph (TOML)
//!
//! ```toml
//! [graph]
//! id = "airspring_eco_pipeline"
//! coordination = "Sequential"
//!
//! [[nodes]]
//! id = "compute_et0"
//! depends_on = ["fetch_weather"]
//! output = "et0_results"
//! capabilities = ["compute"]
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use serde::Deserialize;

/// A parsed `biomeOS` graph definition.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphDef {
    /// Graph-level metadata.
    pub graph: GraphMeta,
    /// Nodes in the graph.
    pub nodes: Vec<NodeDef>,
}

/// Graph-level metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphMeta {
    /// Unique graph identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Coordination strategy: "Sequential" or "Parallel".
    #[serde(default = "default_coordination")]
    pub coordination: String,
}

fn default_coordination() -> String {
    "Sequential".to_string()
}

/// A single node in the graph.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeDef {
    /// Unique node identifier within this graph.
    pub id: String,
    /// IDs of nodes this node depends on.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Output key produced by this node.
    #[serde(default)]
    pub output: Option<String>,
    /// Required capabilities (e.g., "compute", "storage").
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Node operation definition.
    pub operation: OperationDef,
    /// Optional constraints.
    #[serde(default)]
    pub constraints: Option<ConstraintsDef>,
}

/// What this node does.
#[derive(Debug, Clone, Deserialize)]
pub struct OperationDef {
    /// Operation name: `capability_call`, `rpc_call`, or `health_check`.
    pub name: String,
    /// Target primal for RPC calls.
    #[serde(default)]
    pub target: Option<String>,
    /// Operation parameters.
    #[serde(default)]
    pub params: HashMap<String, toml::Value>,
}

/// Execution constraints.
#[derive(Debug, Clone, Deserialize)]
pub struct ConstraintsDef {
    /// Timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Error type for graph operations.
#[derive(Debug)]
pub enum GraphError {
    /// TOML parsing failed.
    Parse(String),
    /// I/O error reading graph file.
    Io(std::io::Error),
    /// Graph has a cycle (not a DAG).
    Cycle(Vec<String>),
    /// Unknown node referenced in `depends_on`.
    UnknownDependency { node: String, dependency: String },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "graph parse error: {msg}"),
            Self::Io(err) => write!(f, "graph I/O error: {err}"),
            Self::Cycle(nodes) => write!(f, "graph cycle detected: {}", nodes.join(" → ")),
            Self::UnknownDependency { node, dependency } => {
                write!(f, "node `{node}` depends on unknown node `{dependency}`")
            }
        }
    }
}

impl std::error::Error for GraphError {}

impl From<std::io::Error> for GraphError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl GraphDef {
    /// Parse a graph from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Parse`] if the TOML is invalid.
    pub fn from_toml(toml_str: &str) -> Result<Self, GraphError> {
        toml::from_str(toml_str).map_err(|e| GraphError::Parse(e.to_string()))
    }

    /// Load a graph from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Io`] or [`GraphError::Parse`].
    pub fn load(path: &Path) -> Result<Self, GraphError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_toml(&contents)
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Validate that all dependencies reference existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::UnknownDependency`] if a dependency is missing.
    pub fn validate_deps(&self) -> Result<(), GraphError> {
        let known: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for node in &self.nodes {
            for dep in &node.depends_on {
                if !known.contains(dep.as_str()) {
                    return Err(GraphError::UnknownDependency {
                        node: node.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Compute topological execution order (Kahn's algorithm).
    ///
    /// Returns node IDs in an order where all dependencies are satisfied
    /// before a node executes.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::Cycle`] if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<&str>, GraphError> {
        self.validate_deps()?;

        let n = self.nodes.len();
        let id_to_idx: HashMap<&str, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.as_str(), i))
            .collect();

        let mut in_degree = vec![0_usize; n];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        for (i, node) in self.nodes.iter().enumerate() {
            for dep in &node.depends_on {
                if let Some(&dep_idx) = id_to_idx.get(dep.as_str()) {
                    adj[dep_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(i, _)| i)
            .collect();

        let mut order = Vec::with_capacity(n);
        while let Some(idx) = queue.pop_front() {
            order.push(self.nodes[idx].id.as_str());
            for &next in &adj[idx] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() == n {
            Ok(order)
        } else {
            let in_cycle: Vec<String> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| in_degree[*i] > 0)
                .map(|(_, n)| n.id.clone())
                .collect();
            Err(GraphError::Cycle(in_cycle))
        }
    }

    /// Count of nodes that use `capability_call` (compute workloads).
    #[must_use]
    pub fn capability_call_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.operation.name == "capability_call")
            .count()
    }

    /// Count of nodes that use `rpc_call` (infrastructure).
    #[must_use]
    pub fn rpc_call_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.operation.name == "rpc_call")
            .count()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    const ECO_PIPELINE: &str = include_str!("../../../graphs/airspring_eco_pipeline.toml");
    const CROSS_PRIMAL: &str = include_str!("../../../graphs/cross_primal_soil_microbiome.toml");

    #[test]
    fn parse_eco_pipeline() {
        let graph = GraphDef::from_toml(ECO_PIPELINE).expect("should parse");
        assert_eq!(graph.graph.id, "airspring_eco_pipeline");
        assert_eq!(graph.graph.coordination, "Sequential");
        assert_eq!(graph.node_count(), 8);
    }

    #[test]
    fn parse_cross_primal() {
        let graph = GraphDef::from_toml(CROSS_PRIMAL).expect("should parse");
        assert_eq!(graph.graph.id, "cross_primal_soil_microbiome");
        assert_eq!(graph.node_count(), 8);
    }

    #[test]
    fn eco_pipeline_topo_order() {
        let graph = GraphDef::from_toml(ECO_PIPELINE).expect("should parse");
        let order = graph.topological_order().expect("should be a DAG");
        assert_eq!(order.len(), 8);

        let pos = |id: &str| order.iter().position(|&n| n == id).unwrap();
        assert!(pos("check_nestgate") < pos("fetch_weather"));
        assert!(pos("check_toadstool") < pos("compute_et0"));
        assert!(pos("fetch_weather") < pos("compute_et0"));
        assert!(pos("compute_et0") < pos("water_balance"));
        assert!(pos("water_balance") < pos("yield_response"));
        assert!(pos("yield_response") < pos("store_results"));
        assert!(pos("store_results") < pos("validate_pipeline"));
    }

    #[test]
    fn cross_primal_topo_order() {
        let graph = GraphDef::from_toml(CROSS_PRIMAL).expect("should parse");
        let order = graph.topological_order().expect("should be a DAG");
        assert_eq!(order.len(), 8);

        let pos = |id: &str| order.iter().position(|&n| n == id).unwrap();
        assert!(pos("soil_moisture") < pos("diversity_analysis"));
        assert!(pos("diversity_analysis") < pos("spectral_analysis"));
    }

    #[test]
    fn capability_call_counts() {
        let eco = GraphDef::from_toml(ECO_PIPELINE).expect("should parse");
        assert_eq!(eco.capability_call_count(), 4);
        assert_eq!(eco.rpc_call_count(), 3);

        let cross = GraphDef::from_toml(CROSS_PRIMAL).expect("should parse");
        assert_eq!(cross.capability_call_count(), 4);
    }

    #[test]
    fn cycle_detection() {
        let cyclic = r#"
[graph]
id = "cyclic"
description = "has cycle"

[[nodes]]
id = "a"
depends_on = ["b"]
[nodes.operation]
name = "test"

[[nodes]]
id = "b"
depends_on = ["a"]
[nodes.operation]
name = "test"
"#;
        let graph = GraphDef::from_toml(cyclic).expect("should parse");
        let result = graph.topological_order();
        assert!(result.is_err());
        if let Err(GraphError::Cycle(nodes)) = result {
            assert_eq!(nodes.len(), 2);
        }
    }

    #[test]
    fn unknown_dependency_detection() {
        let bad_dep = r#"
[graph]
id = "bad"
description = "unknown dep"

[[nodes]]
id = "a"
depends_on = ["nonexistent"]
[nodes.operation]
name = "test"
"#;
        let graph = GraphDef::from_toml(bad_dep).expect("should parse");
        let result = graph.validate_deps();
        assert!(result.is_err());
    }

    #[test]
    fn parallel_roots() {
        let parallel = r#"
[graph]
id = "parallel_roots"
description = "two independent roots"

[[nodes]]
id = "a"
[nodes.operation]
name = "test"

[[nodes]]
id = "b"
[nodes.operation]
name = "test"

[[nodes]]
id = "c"
depends_on = ["a", "b"]
[nodes.operation]
name = "test"
"#;
        let graph = GraphDef::from_toml(parallel).expect("should parse");
        let order = graph.topological_order().expect("should be DAG");
        assert_eq!(order.len(), 3);
        let pos = |id: &str| order.iter().position(|&n| n == id).unwrap();
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("c"));
    }
}
