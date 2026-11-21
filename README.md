# Sokoban solver — make.fun entry

A push-optimal Sokoban solver for `sokoban-solver-01-run-20261121`.

## Approach

The generator builds each level as a one-cell-wide hallway with one to four
rooms off it. A box pushed out of its room through the door can never come
back, so every box stays in its room and a level's push-optimal count is the
sum of its rooms' optima. This entry therefore:

1. splits each level into its rooms (connected floor components off the
   hallway row),
2. solves each room push-optimally with A* over push states — a state is the
   sorted box cells plus the player's reachable region, the heuristic is the
   cheapest assignment of boxes to goals under per-goal push distances, and
   dead squares plus 2x2 wall/box blocks prune dead positions,
3. stitches the per-room move sequences together with walks along the
   hallway, and replays the whole solution to check it before writing it.

The search is budgeted by expanded nodes only. The program never reads the
wall clock, so two runs on one set write byte-identical output.

## Run

```
node main.ts
```

It reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt` (moves in `u`, `d`, `l`, `r`). For a local run
against the kit's practice set:

```
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, no native code.
