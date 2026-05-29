# After Action Report: Wave 60 Eukaryotic Unicellular Deployment — airSpring on eastGate

**Date**: 2026-05-29 (Wave 60)
**Gate**: eastGate (192.168.1.144)
**Spring**: airSpring v0.10.0
**Scenario**: Fresh-gate deployment via Forgejo periplasm + cascade-pull, shared NUCLEUS with groundSpring (parallel IDE), primalSpring dev NUCLEUS co-resident
**Auditor**: airSpring agent (for primalSpring review)

---

## Executive Summary

Treated eastGate as a "somewhat fresh gate" to stress-test the new Wave 60 eukaryotic sync pattern — VPS-based Forgejo as the single source of truth, `cascade-pull.sh` for gate-aware repo sync, and `plasmidBin` for binary deployment. The exercise exposed **8 blocking or near-blocking issues** and **6 improvement opportunities** across the cascade-pull, binary deployment, and multi-tenant coordination layers.

**Bottom line**: The pattern works but was not yet turnkey at the time of testing. A truly fresh gate would have stalled at missing clones, dangling symlinks, and hostname detection. **UPDATE (Wave 60 PM):** Upstream primalSpring addressed 3 of our P0 findings — `cascade-pull.sh` is now manifest-driven (`ecosystem_manifest.toml`), supports `--clone-missing`, and reads `.gate` identity files. Re-tested: 36/38 repos synced (2 known merge conflicts). The symlink and shared-target issues remain open.

---

## 1. What Worked Well

### 1.1 Forgejo SSH Authentication — Zero Friction
- SSH key auth (`eastGate` key) worked on first attempt.
- `git ls-remote`, `git clone`, `git push`, `git pull` all succeeded without prompts or credential issues.
- Round-trip verified: `HEAD == forgejo/main` after push — no drift.
- **Verdict**: SSH key provisioning on the golgiBody VPS is solid.

### 1.2 cascade-pull.sh — Core Loop Is Sound
- Gate profiles are well-structured: eastGate correctly maps to 30 repos across `infra/`, `primals/`, `springs/`, `gardens/`.
- `--ff-only` pull strategy is correct — it prevents silent merge disasters.
- Dry-run mode works correctly.
- The script pulled **25/30 repos** successfully in one pass (~4 seconds total).
- Source selection (`--source forgejo` vs `origin`) works cleanly.
- **Verdict**: The fundamental design is right. Issues are at the edges.

### 1.3 plasmidBin Binary Inventory — Comprehensive
- 15 musl-static binaries present in `primals/x86_64-unknown-linux-musl/`.
- All 13 NUCLEUS primals + `primalspring_primal` + `sourdough` available.
- `ports.env` is well-organized with canonical port assignments, composition profiles (`COMP_TOWER`, `COMP_NODE`, `COMP_NEST`, `COMP_NUCLEUS`, `COMP_FULL`), and per-spring niche profiles.
- `nucleus_launcher.sh` dependency-aware startup order is correct (beardog first, petaltongue last).
- **Verdict**: plasmidBin is the most mature part of the pipeline.

### 1.4 Remote Configuration — Dual-Remote Convention
- All 27 locally-cloned repos have both `origin` (GitHub) and `forgejo` remotes configured.
- `--ensure-remotes` flag in cascade-pull.sh can add forgejo remotes to existing clones.
- Forgejo URL mapping (`forgejo_url()`) correctly handles the org split (ecoPrimals, syntheticChemistry, sporeGarden).
- **Verdict**: The dual-remote convention is robust and well-thought-out.

### 1.5 Gate Profile System — Good Filtering
- eastGate profile correctly includes all springs, all primals, key infra, and relevant gardens.
- Composition definitions in `ports.env` align with the atomic model (Tower/Node/Nest/NUCLEUS).
- Niche profiles (`NICHE_AIRSPRING`, `NICHE_GROUNDSPRING`) correctly define per-spring primal requirements.
- **Verdict**: Gate-awareness is a genuine improvement over the old "clone everything" approach.

---

## 2. What Did Not Work

### 2.1 CRITICAL: Hostname Auto-Detection Fails

**Problem**: `detect_gate()` uses `hostname -s` to determine gate identity. This machine's hostname is `pop-os`, which doesn't match any gate pattern (`east*`, `iron*`, etc.). Running `cascade-pull.sh --gate auto` would produce:
```
WARNING: cannot auto-detect gate from hostname 'pop-os'
```

**Impact**: Every invocation requires explicit `--gate eastGate` or `GATE_NAME=eastGate` environment variable.

**Recommendation**: Add a persistent gate identity file:
```bash
# ~/.config/ecoPrimals/gate.conf or $ECOPRIMALS_ROOT/.gate
GATE_NAME=eastGate
```
Fall back to hostname matching only if no config file exists. Consider also checking `/etc/machine-info` or a DNS TXT record.

### 2.2 CRITICAL: cascade-pull Cannot Clone Missing Repos

**Problem**: 3 repos in the eastGate profile were not cloned locally:
- `primals/songBird` — on Forgejo, needed manual `git clone`
- `primals/nestGate` — on Forgejo, needed manual `git clone`
- `gardens/foundation` — NOT on Forgejo (`sporeGarden/foundation` returns 404), fell back to GitHub

The script prints `SKIP (not cloned)` but takes no remedial action.

**Impact**: A truly fresh gate would have **zero** repos cloned. cascade-pull would skip all 30 and report success.

**Recommendation**: Add `--clone-missing` flag (or make it default behavior):
```bash
if [[ ! -d "$local_path/.git" ]]; then
    url=$(forgejo_url "$repo_path")
    if [[ -n "$url" ]] && $CLONE_MISSING; then
        mkdir -p "$(dirname "$local_path")"
        git clone "$url" "$local_path"
    fi
fi
```

### 2.3 CRITICAL: ECOPRIMALS_ROOT Detection Fragile

**Problem**: When run via symlink (`wateringHole/cascade-pull.sh` → `scripts/cascade-pull.sh`), the `SCRIPT_DIR` resolves to the symlink target's directory, and the 3-level parent walk (`../../..`) may not resolve to the ecoPrimals root depending on invocation context.

**Impact**: First run failed with `ERROR: cannot find ecoPrimals root`. Required manual `ECOPRIMALS_ROOT=/home/eastgate/Development/ecoPrimals`.

**Recommendation**: Use `git rev-parse --show-toplevel` as a fallback, or search upward for a sentinel file (e.g., `ecoPrimals.toml` or `.ecoprimals-root`).

### 2.4 HIGH: Workspace Target Directory Mismatch — Dangling Symlinks

**Problem**: The ecoPrimals-root `.cargo/config.toml` sets `target-dir = "target"`, which resolves to `/home/eastgate/Development/ecoPrimals/target/` (workspace root), NOT `springs/airSpring/barracuda/target/`. Previously created symlinks in plasmidBin pointed to `barracuda/target/release/airspring_primal` — a path that never existed.

**Impact**: `airspring_primal` symlink was **dangling**. Binary built successfully but was not discoverable via the symlink. Any process trying to launch airSpring via plasmidBin would fail.

**Evidence**: `cargo build --release` output: `Finished release target(s) in 17.92s`, but `ls barracuda/target/release/airspring_primal` returned "No such file or directory". Actual binary was at `/home/eastgate/Development/ecoPrimals/target/release/airspring_primal`.

**Recommendation**:
1. Document the canonical binary location: `$ECOPRIMALS_ROOT/target/release/<binary>` (not `crate/target/release/`).
2. Add a `plasmidBin/link_spring.sh` helper that resolves the correct target path automatically:
   ```bash
   ACTUAL=$(cargo metadata --format-version 1 | jq -r '.target_directory')/release/$BIN
   ln -sf "$ACTUAL" "$PLASMIDBIN/primals/$BIN"
   ```
3. Consider copying binaries into plasmidBin instead of symlinking, to survive `cargo clean`.

### 2.5 HIGH: Shared Target Directory — Co-Tenant Contamination

**Problem**: The workspace-level `target/` is shared across ALL crates in the ecoPrimals workspace. A `cargo clean` from any spring or primal wipes binaries for **all** springs and primals, including airSpring's.

**Evidence**: Between Wave 60's first build and the AAR investigation (< 10 minutes), the `target/release/airspring_primal` binary disappeared — likely cleaned by a co-tenant agent or process.

**Impact**: On a shared gate (airSpring + groundSpring + primalSpring), any team running `cargo clean` breaks all other teams' binaries.

**Recommendation**:
1. Copy (not symlink) spring binaries into plasmidBin after build.
2. Or: use per-crate `CARGO_TARGET_DIR` overrides for spring builds.
3. Or: add a `plasmidBin/rebuild.sh` that builds + copies in one atomic operation.

### 2.6 MEDIUM: Merge Conflicts Block cascade-pull

**Problem**: 2 repos (`springs/wetSpring`, `springs/healthSpring`) failed because they had local changes that couldn't fast-forward merge.

**Impact**: `FAILED (try manual merge)` message gives no actionable guidance.

**Recommendation**:
1. After failure, print the conflicting branch state: `git -C "$local_path" log --oneline HEAD..forgejo/main | head -5`
2. Suggest concrete resolution: `cd $repo && git stash && git pull forgejo main && git stash pop`
3. Consider a `--stash-and-pull` mode for automated resolution.

### 2.7 MEDIUM: Dirty Repos Across the Gate

**Problem**: 8 of 27 repos had uncommitted local changes:

| Repo | Dirty Files | Nature |
|------|-------------|--------|
| `infra/wateringHole` | 1 | Modified README |
| `primals/bingoCube` | 1 | Modified Cargo.toml |
| `springs/primalSpring` | 9 | Active development (graphs, validation, specs) |
| `springs/wetSpring` | 1 | Modified CONTEXT.md |
| `springs/neuralSpring` | 1 | Modified CONTEXT.md |
| `springs/healthSpring` | 1 | Modified CONTEXT.md |
| `springs/groundSpring` | 1 | Modified CONTEXT.md |
| `springs/ludoSpring` | 1 | Modified CONTEXT.md |
| `infra/whitePaper` | 13 | neuralAPI chapters in progress |

**Impact**: Dirty state suggests these springs have uncommitted Wave 50-60 work from parallel agents. On a fresh gate, this wouldn't be an issue, but on a lived-in gate it means cascade-pull may conflict.

**Recommendation**: Add `--status` mode to cascade-pull that reports dirty/behind/ahead state for all profile repos without pulling.

### 2.8 LOW: skunkBat Missing from CORE Profile

**Problem**: `primals/skunkBat` is defined in `COMP_TOWER` (the trust boundary electron) and is part of every composition, but it's NOT in the `CORE` variable used to build gate profiles. It only appears in the `golgiBody` profile.

**Impact**: Gates that use `CORE + extras` (eastGate, ironGate, etc.) will still pull skunkBat only if it happens to be listed explicitly. Currently eastGate works because skunkBat is listed in the x86_64-musl binaries, but a new gate profile might miss it.

**Recommendation**: Add `primals/skunkBat` to `CORE`.

---

## 3. Forgejo Organization Gaps

| Repo | Forgejo Org | Status |
|------|-------------|--------|
| `gardens/foundation` | `sporeGarden` | **404 — not on Forgejo** |
| `primals/songBird` | `ecoPrimals` | Present, cloned successfully |
| `primals/nestGate` | `ecoPrimals` | Present, cloned successfully |

**Recommendation**: Mirror `sporeGarden/foundation` to Forgejo, or document it as GitHub-only with a fallback in cascade-pull's `forgejo_url()`.

---

## 4. Multi-Tenant Coordination (eastGate)

### 4.1 Current Tenants
- **airSpring** — ecology science (this agent)
- **groundSpring** — parallel IDE agent, shares NUCLEUS
- **primalSpring** — coordinator, separate dev NUCLEUS

### 4.2 Shared Resources
- `/home/eastgate/Development/ecoPrimals/target/` — workspace build cache (shared, fragile)
- `/home/eastgate/Development/ecoPrimals/infra/plasmidBin/primals/` — binary depot (symlinks into shared target)
- `/run/user/1000/biomeos/` — UDS sockets (NUCLEUS runtime)
- Songbird TCP `:7700` — federation port (shared)

### 4.3 Coordination Gaps
1. No lock file or lease mechanism for `cargo build` — concurrent builds from different springs may race.
2. No per-spring target directory isolation — groundSpring's `cargo clean` will wipe airSpring's binary.
3. No mechanism to signal "NUCLEUS is mine" vs "NUCLEUS is shared" — both groundSpring and airSpring assume they can start/stop NUCLEUS.
4. CONTEXT.md lists co-residents but there's no machine-readable tenant registry.

### 4.4 Recommendation
Create `$ECOPRIMALS_ROOT/.gate/tenants.toml`:
```toml
[gate]
name = "eastGate"
ip = "192.168.1.144"

[tenants.airSpring]
agent = "cursor"
nucleus_role = "consumer"

[tenants.groundSpring]
agent = "cursor-parallel"
nucleus_role = "consumer"

[tenants.primalSpring]
agent = "coordinator"
nucleus_role = "owner"
```

---

## 5. Fresh Gate Deployment Checklist (Proposed)

Based on this exercise, a truly fresh gate deployment would require:

| Step | Tool | Status Today |
|------|------|-------------|
| 1. Set gate identity | Manual (`GATE_NAME=` env) | No auto-detection for non-standard hostnames |
| 2. Clone all repos | Manual (`git clone` x30) | cascade-pull can't clone, only pull |
| 3. Add forgejo remotes | `cascade-pull.sh --ensure-remotes` | Works, but requires repos to exist first |
| 4. Pull latest from Forgejo | `cascade-pull.sh --gate X --source forgejo` | Works for existing clones |
| 5. Build spring binary | `cargo build --release --features local --bin X` | Works, but target path is non-obvious |
| 6. Link binary to plasmidBin | Manual `ln -s` | Symlink target path is wrong in docs |
| 7. Start NUCLEUS | `nucleus_launcher.sh --family-id X` | Works from plasmidBin |
| 8. Deploy spring cell | `cell_launcher.sh <spring> start` | Works if binary symlink is valid |
| 9. Verify federation | `curl :7700/jsonrpc` | Works |
| 10. Push to Forgejo | `git push forgejo main` | Works |

**Proposed improvement**: A single `gate-bootstrap.sh` that runs steps 1-6 automatically:
```bash
gate-bootstrap.sh --gate eastGate --source forgejo --springs airSpring,groundSpring
```

---

## 6. cascade-pull.sh Enhancement Requests

| Priority | Enhancement | Effort |
|----------|-------------|--------|
| P0 | `--clone-missing`: clone repos that aren't local | Small |
| P0 | Gate identity file (`$ECOPRIMALS_ROOT/.gate`) | Small |
| P0 | Add `primals/skunkBat` to CORE | Trivial |
| P1 | `--status` mode: report dirty/behind/ahead without pulling | Medium |
| P1 | Post-failure guidance (stash commands, conflict details) | Small |
| P1 | Mirror `gardens/foundation` to Forgejo | Infra |
| P2 | `--stash-and-pull` for automated dirty-state handling | Medium |
| P2 | `link_spring.sh` helper for correct symlink creation | Small |
| P3 | Machine-readable tenant registry | Design needed |

---

## 7. Metrics

| Metric | Value |
|--------|-------|
| Repos in eastGate profile | 30 |
| Successfully pulled | 25 |
| Cloned manually (were missing) | 3 (`songBird`, `nestGate`, `foundation`) |
| Failed (merge conflict) | 2 (`wetSpring`, `healthSpring`) |
| Dirty repos on gate | 8 (30 total dirty files) |
| plasmidBin musl binaries | 15 |
| Symlink failures discovered | 1 (dangling, wrong target path) |
| Target rebuilds required | 2 (target cleaned by co-tenant between builds) |
| Forgejo round-trip verified | Yes (push + pull, HEAD == forgejo/main) |
| Forgejo SSH auth issues | 0 |
| Total time for cascade-pull | ~4 seconds |
| Total time for binary rebuild | ~18 seconds |
| Blocking issues for fresh gate | 3 (clone, hostname, symlink) |

---

## 8. Conclusion

The Wave 60 eukaryotic pattern is a significant improvement over ad-hoc git linkages. The VPS periplasm (Forgejo) provides a reliable sync point, and cascade-pull's gate profiles correctly filter the ecosystem. However, the tooling assumes repos already exist locally and hostnames match gate names — both false for a fresh deployment.

The three highest-impact fixes are:
1. **Auto-clone missing repos** in cascade-pull
2. **Gate identity file** instead of hostname matching
3. **Copy binaries to plasmidBin** instead of symlinking into a shared target directory

With these three changes, a fresh gate could go from zero to running NUCLEUS in under 5 minutes.

---

*Filed by airSpring agent for primalSpring audit — Wave 60, 2026-05-29*
