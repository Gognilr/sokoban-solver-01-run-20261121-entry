// Room-by-room push-optimal Sokoban solver.
//
// Idea (my own design): the generator lays a one-cell-wide hallway across the
// board with one to four rooms off it. A box pushed out of its room through
// the door can never come back, so every box stays in its room, and a level's
// push-optimal count is the sum of its rooms' optima. So each room is solved
// on its own, push-optimally, with A* over push states, and the per-room move
// sequences are stitched together with plain walks along the hallway.
//
// A push state is the sorted box cells plus the player's reachable region
// (every cell of the region is equivalent for future pushes, so the region's
// lowest cell identifies it). The heuristic is the cheapest assignment of
// boxes to goals under per-goal push distances (a box's fewest pushes to a
// goal ignoring the other boxes), which is admissible and consistent, so the
// first goal state A* pops is push-optimal. Dead positions are pruned with
// static dead squares (no push path to any goal) and 2x2 blocks of walls and
// boxes. The search is budgeted by expanded nodes only: this program never
// reads the wall clock, so two runs on one set write byte-identical output.
//
// Node 22 erasable TypeScript only, standard library only, no network.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets, counted in nodes, never in time.
const FIRST_CAP = 200_000;
const RETRY_CAP = 800_000;
const TOTAL_CAP = 3_000_000;

const INF = 1_000_000_000;

const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
] as const;

interface Budget {
  remaining: number;
}

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  cells: Uint8Array; // room floor cells (walls excluded)
  boxes: number[];
  goals: number[];
  doors: number[]; // room cells touching the hallway
  entries: number[]; // hallway cells touching a door cell
}

interface Push {
  box: number;
  dir: number; // index into DIRS
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const c = y * w + x;
      if (ch !== '#') wall[c] = 0;
      if ('.*+'.includes(ch)) goals.push(c);
      if ('$*'.includes(ch)) boxes.push(c);
      if ('@+'.includes(ch)) player = c;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Split the level into the hallway row (the row the player starts on) and
// the rooms hanging off it. A room is a connected component of floor cells
// outside the hallway row.
function findRooms(level: Level): { hall: Uint8Array; rooms: Room[] } {
  const { w, h, wall } = level;
  const hallRow = Math.floor(level.player / w);
  const hall = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) {
    const c = hallRow * w + x;
    if (!wall[c]) hall[c] = 1;
  }
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  const offs = [1, -1, w, -w];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x;
      if (wall[start] || hall[start] || seen[start]) continue;
      const comp: number[] = [];
      const stack = [start];
      seen[start] = 1;
      while (stack.length > 0) {
        const c = stack.pop()!;
        comp.push(c);
        for (const o of offs) {
          const n = c + o;
          if (n < 0 || n >= w * h) continue;
          if (wall[n] || hall[n] || seen[n]) continue;
          // Do not wrap around row edges.
          if (o === 1 && n % w === 0) continue;
          if (o === -1 && c % w === 0) continue;
          seen[n] = 1;
          stack.push(n);
        }
      }
      const cells = new Uint8Array(w * h);
      for (const c of comp) cells[c] = 1;
      const boxes = level.boxes.filter((b) => cells[b]);
      const goals = level.goals.filter((g) => cells[g]);
      const doors: number[] = [];
      const entries: number[] = [];
      for (const c of comp) {
        for (const o of offs) {
          const n = c + o;
          if (n < 0 || n >= w * h) continue;
          if (o === 1 && n % w === 0) continue;
          if (o === -1 && c % w === 0) continue;
          if (hall[n]) {
            doors.push(c);
            entries.push(n);
          }
        }
      }
      doors.sort((a, b) => a - b);
      entries.sort((a, b) => a - b);
      rooms.push({ cells, boxes, goals, doors, entries });
    }
  }
  rooms.sort((a, b) => (a.doors[0] ?? 0) - (b.doors[0] ?? 0));
  return { hall, rooms };
}

// BFS walk from `from` to `to` over cells where allowed[c] is set and no box
// sits. Returns the move string, or null when unreachable. Neighbours are
// tried in the fixed order u, d, l, r, so the result is deterministic.
function walkMoves(
  level: Level,
  allowed: Uint8Array,
  boxes: number[],
  from: number,
  to: number,
): string | null {
  const { w, h, wall } = level;
  if (from === to) return '';
  const blocked = new Uint8Array(w * h);
  for (const b of boxes) blocked[b] = 1;
  const prev = new Int32Array(w * h).fill(-1);
  const prevDir = new Int8Array(w * h).fill(-1);
  const queue = [from];
  prev[from] = from;
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++]!;
    for (let d = 0; d < 4; d++) {
      const dir = DIRS[d]!;
      const n = c + dir.dy * w + dir.dx;
      if (n < 0 || n >= w * h) continue;
      if (dir.dx === 1 && n % w === 0) continue;
      if (dir.dx === -1 && c % w === 0) continue;
      if (wall[n] || !allowed[n] || blocked[n] || prev[n] !== -1) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let moves = '';
        let cur = to;
        while (cur !== from) {
          moves = DIRS[prevDir[cur]!]!.ch + moves;
          cur = prev[cur]!;
        }
        return moves;
      }
      queue.push(n);
    }
  }
  return null;
}

// Flood fill of the player's reachable region from `start` over passable
// cells (room or hallway) that hold no box. Returns the region mask and its
// lowest cell, which identifies the region.
function floodRegion(
  passable: Uint8Array,
  boxMark: Uint8Array,
  start: number,
  w: number,
  h: number,
): { mask: Uint8Array; low: number } {
  const mask = new Uint8Array(w * h);
  const stack = [start];
  mask[start] = 1;
  let low = start;
  while (stack.length > 0) {
    const c = stack.pop()!;
    if (c < low) low = c;
    const x = c % w;
    if (x + 1 < w) {
      const n = c + 1;
      if (passable[n] && !boxMark[n] && !mask[n]) {
        mask[n] = 1;
        stack.push(n);
      }
    }
    if (x > 0) {
      const n = c - 1;
      if (passable[n] && !boxMark[n] && !mask[n]) {
        mask[n] = 1;
        stack.push(n);
      }
    }
    const n2 = c + w;
    if (n2 < w * h && passable[n2] && !boxMark[n2] && !mask[n2]) {
      mask[n2] = 1;
      stack.push(n2);
    }
    const n3 = c - w;
    if (n3 >= 0 && passable[n3] && !boxMark[n3] && !mask[n3]) {
      mask[n3] = 1;
      stack.push(n3);
    }
  }
  return { mask, low };
}

interface RoomSolution {
  pushes: Push[];
  expansions: number;
  finalBoxes: number[];
  finalPlayer: number;
}

// Push-optimal A* for one room. The player starts on an entry cell (a
// hallway cell beside the door); during the solve the player may roam the
// room and the hallway, but a box may only ever be pushed onto a room cell.
// Returns null when the node budget runs out.
function solveRoom(
  level: Level,
  hall: Uint8Array,
  room: Room,
  entry: number,
  budget: Budget,
  cap: number,
): RoomSolution | null {
  const { w, h, wall } = level;
  const n = room.boxes.length;
  const goals = room.goals;
  if (n === 0) {
    return { pushes: [], expansions: 0, finalBoxes: [], finalPlayer: entry };
  }
  if (n !== goals.length) return null;

  // Passable cells for the player: the room plus the whole hallway (no box
  // ever sits in the hallway while this room is solved, so the hallway is
  // one freely walkable stretch that joins every entry cell).
  const passable = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) passable[c] = room.cells[c] || hall[c] ? 1 : 0;

  const offs = DIRS.map((d) => d.dy * w + d.dx);

  // Per-goal push distances: dist[gi][c] is the fewest pushes that move a
  // box from c onto goals[gi], ignoring the other boxes. Reverse BFS: a box
  // at p pushed in direction d lands on c = p + d, which needs the player
  // to stand on p - d, a passable cell.
  const dist: Int32Array[] = [];
  for (let gi = 0; gi < n; gi++) {
    const d = new Int32Array(w * h).fill(INF);
    const g = goals[gi]!;
    d[g] = 0;
    const queue = [g];
    let head = 0;
    while (head < queue.length) {
      const c = queue[head++]!;
      for (let k = 0; k < 4; k++) {
        const dir = DIRS[k]!;
        const p = c - offs[k]!;
        if (p < 0 || p >= w * h || !room.cells[p] || d[p] !== INF) continue;
        // Horizontal steps must stay on the same row (no wraparound).
        if (dir.dx !== 0 && Math.floor(p / w) !== Math.floor(c / w)) continue;
        const stand = p - offs[k]!;
        if (stand < 0 || stand >= w * h || !passable[stand]) continue;
        if (dir.dx !== 0 && Math.floor(stand / w) !== Math.floor(p / w)) continue;
        d[p] = d[c]! + 1;
        queue.push(p);
      }
    }
    dist.push(d);
  }
  // distMat[c * n + gi]: pushes from cell c to goal gi.
  const distMat = new Int32Array(w * h * n);
  for (let c = 0; c < w * h; c++) {
    for (let gi = 0; gi < n; gi++) distMat[c * n + gi] = dist[gi]![c]!;
  }
  const deadCell = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) {
    if (!room.cells[c]) continue;
    let live = false;
    for (let gi = 0; gi < n; gi++) {
      if (distMat[c * n + gi] < INF) {
        live = true;
        break;
      }
    }
    if (!live) deadCell[c] = 1;
  }

  // 2x2 windows that can trap boxes: every cell wall-or-room, at least one
  // a room cell (a hallway cell in the window means the window can never be
  // full, since no box ever enters the hallway).
  const windows: number[][] = [];
  for (let y = 0; y + 1 < h; y++) {
    for (let x = 0; x + 1 < w; x++) {
      const quad = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
      if (!quad.some((c) => room.cells[c])) continue;
      if (quad.some((c) => !wall[c] && !room.cells[c])) continue;
      windows.push(quad);
    }
  }
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  // isDead uses its own scratch marks so it never disturbs the caller's
  // box marks.
  const scratch = new Uint8Array(w * h);
  const isDead = (boxes: number[]): boolean => {
    for (const b of boxes) if (deadCell[b]) return true;
    for (const b of boxes) scratch[b] = 1;
    let dead = false;
    for (const quad of windows) {
      if (!quad.every((c) => wall[c] || scratch[c])) continue;
      if (quad.some((c) => scratch[c] && !goalSet[c])) {
        dead = true;
        break;
      }
    }
    for (const b of boxes) scratch[b] = 0;
    return dead;
  };

  // All permutations of goal indices, for the assignment heuristic.
  const perms: number[][] = [];
  const buildPerms = (prefix: number[], rest: number[]): void => {
    if (rest.length === 0) {
      perms.push(prefix);
      return;
    }
    for (let i = 0; i < rest.length; i++) {
      buildPerms(
        [...prefix, rest[i]!],
        [...rest.slice(0, i), ...rest.slice(i + 1)],
      );
    }
  };
  buildPerms([], goals.map((_, i) => i));

  const heuristic = (boxes: number[]): number => {
    let best = INF;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = distMat[boxes[i]! * n + perm[i]!]!;
        if (d >= INF) {
          s = INF;
          break;
        }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  // Numeric state keys: boxes packed base-(w*h), times (w*h), plus the
  // region's lowest cell. w*h is at most a few hundred, so the key stays
  // far below 2^53.
  const B = w * h;
  const stateKey = (boxes: number[], low: number): number => {
    let k = 0;
    for (let i = n - 1; i >= 0; i--) k = k * B + boxes[i]!;
    return k * B + low;
  };

  // A* state storage.
  const sBoxes: number[][] = [];
  const sLow: number[] = [];
  const sRegion: (Uint8Array | null)[] = [];
  const sG: number[] = [];
  const sF: number[] = [];
  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const sKey: number[] = [];
  const best = new Map<number, number>(); // state key -> smallest g seen

  // Binary heap of state ids, ordered by (f, larger g first, older first).
  const heap: number[] = [];
  let seq = 0;
  const sSeq: number[] = [];
  const heapLess = (a: number, b: number): boolean => {
    if (sF[a] !== sF[b]) return sF[a]! < sF[b]!;
    if (sG[a] !== sG[b]) return sG[a]! > sG[b]!;
    return sSeq[a]! < sSeq[b]!;
  };
  const heapPush = (id: number): void => {
    let i = heap.length;
    heap.push(id);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const addState = (
    boxes: number[],
    low: number,
    region: Uint8Array,
    g: number,
    h: number,
    parent: number,
    pushBox: number,
    pushDir: number,
  ): number => {
    const id = sBoxes.length;
    sBoxes.push(boxes);
    sLow.push(low);
    sRegion.push(region);
    sG.push(g);
    sF.push(g + h);
    sParent.push(parent);
    sPushBox.push(pushBox);
    sPushDir.push(pushDir);
    sKey.push(stateKey(boxes, low));
    sSeq.push(seq++);
    return id;
  };

  const startBoxes = room.boxes.slice().sort((a, b) => a - b);
  if (isDead(startBoxes)) return null;
  const boxMark = new Uint8Array(w * h);
  for (const b of startBoxes) boxMark[b] = 1;
  const startRegion = floodRegion(passable, boxMark, entry, w, h);
  for (const b of startBoxes) boxMark[b] = 0;
  const startId = addState(startBoxes, startRegion.low, startRegion.mask, 0, heuristic(startBoxes), -1, -1, -1);
  best.set(sKey[startId]!, 0);
  heapPush(startId);

  let expansions = 0;
  while (heap.length > 0) {
    const id = heapPop();
    if (best.get(sKey[id]!) !== sG[id]) continue; // stale duplicate
    const boxes = sBoxes[id]!;
    if (boxes.every((b) => goalSet[b])) {
      const pushes: Push[] = [];
      let cur = id;
      while (sParent[cur]! >= 0) {
        pushes.push({ box: sPushBox[cur]!, dir: sPushDir[cur]! });
        cur = sParent[cur]!;
      }
      pushes.reverse();
      return {
        pushes,
        expansions,
        finalBoxes: boxes,
        finalPlayer: sPushBox[id]! >= 0 ? sPushBox[id]! : entry,
      };
    }
    if (expansions >= cap || budget.remaining <= 0) return null;
    expansions++;
    budget.remaining--;
    const region = sRegion[id]!;
    sRegion[id] = null; // free the mask once expanded
    for (const b of boxes) boxMark[b] = 1;
    const g2 = sG[id]! + 1;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      for (let d = 0; d < 4; d++) {
        const dir = DIRS[d]!;
        if (dir.dx === 1 && bx === w - 1) continue;
        if (dir.dx === -1 && bx === 0) continue;
        const t = b + offs[d]!;
        if (t < 0 || t >= w * h || !room.cells[t] || boxMark[t]) continue;
        if (dir.dx === 1 && bx === 0) continue; // behind cell would wrap
        if (dir.dx === -1 && bx === w - 1) continue;
        const behind = b - offs[d]!;
        if (behind < 0 || behind >= w * h || !region[behind]) continue;
        const nb = boxes.slice();
        nb[bi] = t;
        nb.sort((a, b2) => a - b2);
        if (isDead(nb)) continue;
        const h2 = heuristic(nb);
        if (h2 >= INF) continue;
        // The new region is flooded with the moved box on t and its old
        // cell b free (the player ends there); swap the marks for the
        // flood, then restore them.
        boxMark[b] = 0;
        boxMark[t] = 1;
        const reg = floodRegion(passable, boxMark, b, w, h);
        boxMark[t] = 0;
        boxMark[b] = 1;
        const key = stateKey(nb, reg.low);
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g2) continue;
        best.set(key, g2);
        const nid = addState(nb, reg.low, reg.mask, g2, h2, id, b, d);
        heapPush(nid);
      }
    }
    for (const b of boxes) boxMark[b] = 0;
  }
  return null;
}

// Replay a room's pushes into concrete moves, starting with the player on
// `entry`. Walks stay inside the room and the hallway.
function replayRoom(
  level: Level,
  hall: Uint8Array,
  room: Room,
  entry: number,
  pushes: Push[],
): { moves: string; finalPlayer: number; finalBoxes: number[] } {
  const { w, h } = level;
  const passable = new Uint8Array(w * h);
  for (let c = 0; c < w * h; c++) passable[c] = room.cells[c] || hall[c] ? 1 : 0;
  const offs = DIRS.map((d) => d.dy * w + d.dx);
  let player = entry;
  let boxes = room.boxes.slice().sort((a, b) => a - b);
  let moves = '';
  for (const p of pushes) {
    const behind = p.box - offs[p.dir]!;
    const walk = walkMoves(level, passable, boxes, player, behind);
    if (walk === null) throw new Error('room replay: push unreachable');
    moves += walk;
    moves += DIRS[p.dir]!.ch;
    boxes[boxes.indexOf(p.box)] = p.box + offs[p.dir]!;
    boxes.sort((a, b) => a - b);
    player = p.box;
  }
  return { moves, finalPlayer: player, finalBoxes: boxes };
}

// Replay a move string on the level; returns the push count, or -1 when the
// replay is illegal or leaves a box off its goals.
function verifyMoves(level: Level, moves: string): number {
  const { w, h, wall } = level;
  const goals = new Set(level.goals);
  const boxes = new Set(level.boxes);
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = DIRS.findIndex((dir) => dir.ch === ch);
    if (d < 0) return -1;
    const dir = DIRS[d]!;
    if (dir.dx === 1 && player % w === w - 1) return -1;
    if (dir.dx === -1 && player % w === 0) return -1;
    const q = player + dir.dy * w + dir.dx;
    if (q < 0 || q >= w * h || wall[q]) return -1;
    if (boxes.has(q)) {
      const t = q + dir.dy * w + dir.dx;
      if (dir.dx === 1 && q % w === w - 1) return -1;
      if (dir.dx === -1 && q % w === 0) return -1;
      if (t < 0 || t >= w * h || wall[t] || boxes.has(t)) return -1;
      boxes.delete(q);
      boxes.add(t);
      pushes++;
    }
    player = q;
  }
  if (boxes.size !== goals.size) return -1;
  for (const b of boxes) if (!goals.has(b)) return -1;
  return pushes;
}

function solveLevel(text: string, budget: Budget): { moves: string; pushes: number; detail: string } {
  const level = parseLevel(text);
  const { hall, rooms } = findRooms(level);
  if (rooms.length === 0) return { moves: '', pushes: -1, detail: 'no rooms found' };

  // Solve every room on its own first; a room's solution does not depend on
  // the order rooms are visited in.
  const solved: { room: Room; entry: number; sol: RoomSolution; replay: { moves: string; finalPlayer: number; finalBoxes: number[] } }[] = [];
  const notes: string[] = [];
  for (const room of rooms) {
    if (room.entries.length === 0 || room.boxes.length !== room.goals.length) {
      return { moves: '', pushes: -1, detail: 'room without door or box/goal mismatch' };
    }
    const entry = room.entries[0]!;
    let sol = solveRoom(level, hall, room, entry, budget, FIRST_CAP);
    if (sol === null && budget.remaining > 0) {
      sol = solveRoom(level, hall, room, entry, budget, RETRY_CAP);
    }
    if (sol === null) {
      return { moves: '', pushes: -1, detail: `room at door ${room.doors[0]} unsolved (budget)` };
    }
    const replay = replayRoom(level, hall, room, entry, sol.pushes);
    notes.push(`room@${room.doors[0]}: ${sol.pushes.length} pushes, ${sol.expansions} expansions`);
    solved.push({ room, entry, sol, replay });
  }

  // A room whose solved state traps the player inside (a box ends on its
  // only way out) has to be visited last; all others can go in door order.
  const canExit = (s: (typeof solved)[number]): boolean => {
    const allowed = new Uint8Array(level.w * level.h);
    for (let c = 0; c < level.w * level.h; c++) allowed[c] = hall[c] || s.room.cells[c] ? 1 : 0;
    return walkMoves(level, allowed, s.replay.finalBoxes, s.replay.finalPlayer, s.entry) !== null;
  };
  const ordered = solved.slice().sort((a, b) => {
    const ta = canExit(a) ? 0 : 1;
    const tb = canExit(b) ? 0 : 1;
    if (ta !== tb) return ta - tb;
    return (a.room.doors[0] ?? 0) - (b.room.doors[0] ?? 0);
  });

  // Stitch: walk the hallway to each room's door, run its moves, walk on.
  const allAllowed = new Uint8Array(level.w * level.h);
  for (let c = 0; c < level.w * level.h; c++) allAllowed[c] = level.wall[c] ? 0 : 1;
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of ordered) {
    const walk = walkMoves(level, allAllowed, boxes, player, s.entry);
    if (walk === null) {
      return { moves: '', pushes: -1, detail: `cannot walk to room at door ${s.room.doors[0]}` };
    }
    moves += walk;
    moves += s.replay.moves;
    player = s.replay.finalPlayer;
    boxes = s.replay.finalBoxes;
  }
  const pushes = verifyMoves(level, moves);
  if (pushes < 0) {
    return { moves: '', pushes: -1, detail: 'stitched solution failed verification' };
  }
  return { moves, pushes, detail: notes.join('; ') };
}

const budget: Budget = { remaining: TOTAL_CAP };
for (const id of readdirSync(I).sort()) {
  const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
  const { moves, pushes, detail } = solveLevel(text, budget);
  console.error(`[${id}] pushes=${pushes} budget_left=${budget.remaining} :: ${detail}`);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
