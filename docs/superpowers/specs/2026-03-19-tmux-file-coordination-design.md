# Tmux File Coordination Design

**Date:** 2026-03-19

## Goal

Build a real controller-worker coordination layer for HannsDB development that does not depend on reading tmux scrollback as the source of truth.

The design must make multi-Codex collaboration observable, resumable, and automatable.

## Problem

The current collaboration pattern is fragile:

- tasks are sent through tmux panes
- workers report completion in pane text
- controller has to remember to re-check panes
- monitor processes can die silently
- old `STATUS: DONE` output and new `Working` output can coexist in the same pane

This means the controller can miss completion and fail to auto-dispatch the next task.

## Design principle

tmux is only the transport layer.

The coordination truth must live in files inside the repository.

## Proposed architecture

### `.coord/` as the coordination source of truth

Add a repository-local coordination directory:

```text
.coord/
  README.md
  events.log
  tasks/
    <task-id>.md
  results/
    <task-id>.md
  workers/
    <worker-id>/
      state.json
```

### State model

Each worker has one state file:

- `worker_id`
- `pane_id`
- `status`
- `task_id`
- `updated_at`
- `note`

Allowed statuses:

- `idle`
- `assigned`
- `acknowledged`
- `working`
- `blocked`
- `done`
- `done_with_concerns`
- `reviewed`

### Task model

Each task is a file in `.coord/tasks/<task-id>.md`.

Each task file contains:

- task id
- owner worker
- pane id
- goal
- allowed files
- forbidden scope
- required verification
- required output location

### Result model

Each completed task writes `.coord/results/<task-id>.md`.

The result file must contain:

- `STATUS`
- `SUMMARY`
- `FILES`
- `VERIFY`
- `NEXT`

This keeps the worker contract stable but moves it out of pane scrollback.

## Script surface

Add `scripts/coord/` with the following minimal scripts:

- `init.sh`
  - bootstrap `.coord/`
  - register workers and panes
- `set-worker-state.sh`
  - update a worker state file
- `dispatch-task.sh`
  - assign task to worker
  - update state to `assigned`
  - send the task execution prompt to the worker pane
  - enforce `sleep 5` before `Enter`
- `status.sh`
  - summarize all worker states and current tasks
- `monitor.sh`
  - poll worker state files at low frequency
  - append only meaningful state changes to `.coord/events.log`
  - emit heartbeat file in future if needed

## Controller workflow

1. Controller creates or edits `.coord/tasks/<task-id>.md`
2. Controller runs `dispatch-task.sh`
3. Worker reads task file
4. Worker updates state:
   - `acknowledged`
   - `working`
   - `blocked` or `done*`
5. Worker writes `.coord/results/<task-id>.md`
6. Controller or monitor sees `done*`
7. Controller reviews result, runs verification, and sets:
   - `reviewed`
8. Controller creates next task

## Worker contract

Worker panes should receive only a short execution instruction, not the full task content inline:

```text
执行 .coord/tasks/T-20260319-001.md
先把 state.json 更新为 acknowledged / working
完成后写 .coord/results/T-20260319-001.md
最后把 state.json 更新为 done 或 done_with_concerns
```

This keeps pane traffic small and deterministic.

## Why this is better

- completion is durable
- controller can resume after interruption
- task ownership is explicit
- worker output is reviewable without scraping scrollback
- monitoring can be restarted safely
- multiple task rounds are distinguishable by `task_id`

## Skill relationship

This mechanism should also be packaged as a Codex skill so future sessions can reuse it.

The skill should teach:

- when to use tmux file-backed coordination
- why tmux scrollback is not enough
- the required `.coord` state machine
- the required dispatch pattern with `sleep 5`
- the result and review contract

The skill should reference the HannsDB repo scripts as the concrete implementation.

## Out of scope

For the first version:

- no web dashboard
- no SQLite task database
- no daemonized queue scheduler
- no automatic merge/conflict resolution
- no task dependency graph beyond controller judgment

## Acceptance criteria

The mechanism is considered real only if:

1. task assignment is file-backed
2. worker state is file-backed
3. result reporting is file-backed
4. controller can detect completion without reading pane scrollback
5. a new session can understand and reuse the mechanism through one skill and one repo-local README
