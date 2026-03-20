# Coordination Directory

`.coord/` is the repository-local source of truth for tmux-based multi-worker coordination.

## Quick start

Initialize workers for the current three-pane layout:

```bash
bash scripts/coord/init.sh --worker worker1:%1 --worker worker2:%2
```

Check current states:

```bash
bash scripts/coord/status.sh
bash scripts/coord/monitor.sh --once
```

Start/stop the background services:

```bash
bash scripts/coord/start-monitor.sh --interval 20
bash scripts/coord/start-watch.sh --interval 20 --notify-session HannsDB
bash scripts/coord/stop-watch.sh
bash scripts/coord/stop-monitor.sh
```

Dispatch a task after writing `.coord/tasks/<task-id>.md`:

```bash
bash scripts/coord/dispatch-task.sh --worker worker1 --task-id T-20260319-001
```

Workers should:

1. update `state.json` to `acknowledged`
2. update `state.json` to `working`
3. write `.coord/results/<task-id>.md`
4. update `state.json` to `done` or `done_with_concerns`

## Layout

- `tasks/<task-id>.md`
  - controller-authored task file
- `results/<task-id>.md`
  - worker-authored result file
- `workers/<worker-id>/state.json`
  - current worker state
- `events.log`
  - append-only state transition log
- `alerts.log`
  - append-only done/done_with_concerns alert log
- `monitor/`
  - monitor snapshots
- `watch/`
  - watch snapshots and heartbeat

## State model

Each worker state file contains:

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

## Rules

- tmux scrollback is not the source of truth
- task ownership must be tied to a `task_id`
- worker completion must be written to `.coord/results/<task-id>.md`
- controller reacts to worker state files, not pane text
