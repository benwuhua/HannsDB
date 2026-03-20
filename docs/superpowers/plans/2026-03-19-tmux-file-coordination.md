# Tmux File Coordination Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repository-local `.coord` coordination layer with tmux dispatch scripts and package the mechanism as a reusable Codex skill.

**Architecture:** `.coord` becomes the coordination source of truth, `scripts/coord` provides the controller/worker shell surface, and a global skill documents the workflow for reuse across sessions.

**Tech Stack:** Bash, tmux, repository-local Markdown/JSON files, Codex skill format

---

### Task 1: Add coordination design artifacts

**Files:**
- Create: `docs/superpowers/specs/2026-03-19-tmux-file-coordination-design.md`
- Create: `docs/superpowers/plans/2026-03-19-tmux-file-coordination.md`

- [ ] Write the design doc describing `.coord`, worker states, tasks, results, and monitor behavior.
- [ ] Write this implementation plan with exact file ownership and test steps.

### Task 2: Write failing coordination smoke test

**Files:**
- Create: `scripts/coord/test-smoke.sh`

- [ ] Write a shell smoke test that expects:
  - `.coord` bootstrap to exist
  - worker state files to be created
  - state update commands to work
  - monitor `--once` mode to emit event lines
  - dispatch `--dry-run` mode to print the tmux command payload
- [ ] Run: `bash scripts/coord/test-smoke.sh`
Expected: fail because scripts and `.coord` helpers do not exist yet.

### Task 3: Implement `.coord` bootstrap and state helpers

**Files:**
- Create: `.coord/README.md`
- Create: `scripts/coord/lib.sh`
- Create: `scripts/coord/init.sh`
- Create: `scripts/coord/set-worker-state.sh`
- Modify: `scripts/coord/test-smoke.sh`

- [ ] Implement repo-root and coord-root helpers.
- [ ] Implement `init.sh` to create `.coord/tasks`, `.coord/results`, `.coord/workers/<id>/state.json`.
- [ ] Implement `set-worker-state.sh` to update state atomically enough for shell usage.
- [ ] Run: `bash scripts/coord/test-smoke.sh`
Expected: still fail on monitor/dispatch portions.

### Task 4: Implement monitor and status views

**Files:**
- Create: `scripts/coord/status.sh`
- Create: `scripts/coord/monitor.sh`
- Modify: `scripts/coord/test-smoke.sh`

- [ ] Add `status.sh` to summarize all worker states.
- [ ] Add `monitor.sh --once` for testability and `monitor.sh --interval N` for normal use.
- [ ] Append state transitions to `.coord/events.log`.
- [ ] Run: `bash scripts/coord/test-smoke.sh`
Expected: fail only on dispatch portion.

### Task 5: Implement tmux dispatch

**Files:**
- Create: `scripts/coord/dispatch-task.sh`
- Modify: `scripts/coord/test-smoke.sh`

- [ ] Implement `dispatch-task.sh` to:
  - validate task file
  - set state to `assigned`
  - send the short execution instruction to the right pane
  - enforce `sleep 5` before `Enter`
- [ ] Add `--dry-run` mode so the smoke test does not need a live worker execution.
- [ ] Run: `bash scripts/coord/test-smoke.sh`
Expected: pass.

### Task 6: Install skill

**Files:**
- Create: `/Users/ryan/.codex/skills/tmux-file-coordination/SKILL.md`
- Optionally create supporting references under `/Users/ryan/.codex/skills/tmux-file-coordination/`

- [ ] Write the skill around the actual failure mode observed in this session:
  - controller missed `DONE`
  - tmux scrollback was not a reliable source of truth
- [ ] Reference the repo-local `.coord` and `scripts/coord` implementation as the concrete workflow.
- [ ] Keep the skill reusable for future multi-pane Codex sessions.

### Task 7: Verify end-to-end and document usage

**Files:**
- Modify: `.coord/README.md`
- Modify: `docs/hannsdb-project-plan.md`

- [ ] Run the smoke test again.
- [ ] Run `scripts/coord/init.sh` against the current repo.
- [ ] Verify `scripts/coord/status.sh` and `scripts/coord/monitor.sh --once`.
- [ ] Add a short usage section explaining how controller and workers should use the new mechanism.
