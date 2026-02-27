// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified device inventory — discovers all available compute substrates.

use crate::neural;
use crate::probe;
use crate::substrate::Substrate;

/// Discover all substrates on this machine.
///
/// Returns CPU (always), plus any GPUs, NPUs, and biomeOS Neural API found.
#[must_use]
pub fn discover() -> Vec<Substrate> {
    let mut substrates = Vec::new();
    substrates.push(probe::probe_cpu());
    substrates.extend(probe::probe_gpus());
    substrates.extend(probe::probe_npus());
    substrates.extend(neural::probe_neural());
    substrates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::SubstrateKind;

    #[test]
    fn always_has_cpu() {
        let inv = discover();
        assert!(
            inv.iter().any(|s| s.kind == SubstrateKind::Cpu),
            "CPU must always be discovered"
        );
    }

    #[test]
    fn no_duplicates() {
        let inv = discover();
        let cpu_count = inv.iter().filter(|s| s.kind == SubstrateKind::Cpu).count();
        assert_eq!(cpu_count, 1, "should have exactly one CPU entry");
    }
}
