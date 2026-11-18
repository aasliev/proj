// Fig: EA888 Gen 2 (CAEB) 2.0 TFSI, a full engine build on a stand.
// Build: node docs/figures/ea888-caeb/build.mjs [--light] [--audit]
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../../.claude/skills/anatomy/kit/iso-kit.mjs";
import * as G from "../../../.claude/skills/anatomy/kit/lathe.mjs";
import { tubePieces, tubeSvg } from "../../../.claude/skills/anatomy/kit/tube.mjs";
import * as A from "../../../.claude/skills/anatomy/kit/audit.mjs";
import { kitScript } from "../../../.claude/skills/anatomy/scripts/inline-kit.mjs";
import { pdfButton } from "../../../.claude/skills/anatomy/kit/pdf-button.mjs";
import { makeModel } from "./model.mjs";
import { makeMoving } from "./moving.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const KIT = join(HERE, "../../../.claude/skills/anatomy/kit");
const THEME = process.argv.includes("--light") ? "light" : "dark";
const M = makeModel();

const W = 1000;
const H = 1060;
const AZIMUTH = 32;
const ELEVATION = 27;
const FLOOR = -560;

// Assemblies in build order. `move` is where each waits before it goes on.
const PIECES = {
  stand: { name: "Engine stand", say: "", move: [0, 0, 0] },
  block: { name: "Cylinder block", say: "cast-iron block · 82.5 mm bores at 88 mm", move: [0, 0, 0] },
  balance: { name: "Balance shafts", say: "balance shafts · twice crank speed", move: [0, 250, -190] },
  crank: { name: "Crankshaft and main caps", say: "crankshaft · 92.8 mm stroke · five mains", move: [0, 0, -170] },
  pistons: { name: "Pistons and rods", say: "pistons and rods · 144 mm rods", move: [0, 300, 60] },
  oilpump: { name: "Oil pump and chain", say: "two-stage oil pump · chain driven", move: [170, 0, -230] },
  head: { name: "Cylinder head and valvetrain", say: "head · 16 valves · roller fingers", move: [0, 0, 270] },
  cams: { name: "Camshafts and adjuster", say: "camshafts · intake adjuster · valvelift on exhaust", move: [0, -190, 300] },
  chains: { name: "Timing and balance chains", say: "chains · cam drive and balance drive", move: [250, 0, 0] },
  tcover: { name: "Timing chain cover", say: "timing chain cover", move: [400, 0, 0] },
  pan: { name: "Oil pan and dipstick", say: "two-piece oil pan", move: [0, 0, -290] },
  cover: { name: "Cylinder head cover", say: "head cover · valvelift actuators · breather", move: [0, 0, 380] },
  fuel: { name: "High-pressure fuel system", say: "fuel · pump on the intake cam · rail · 4 injectors", move: [0, -220, 220] },
  intake: { name: "Intake manifold and throttle", say: "intake manifold · flaps · throttle body", move: [0, -320, 120] },
  turbo: { name: "Turbocharger and exhaust manifold", say: "turbocharger · wastegate · oil and coolant lines", move: [-40, 360, -170] },
  cooling: { name: "Oil filter and coolant module", say: "oil filter and coolant module", move: [380, -160, -400] },
  belt: { name: "Belt drive and accessories", say: "belt drive · alternator · A/C compressor", move: [430, -220, 0] },
  ignition: { name: "Coils and wiring harness", say: "coil packs and harness · ready to run", move: [0, 0, 460] },
};
const BUILD = ["block", "balance", "crank", "pistons", "oilpump", "head", "cams", "chains", "tcover", "pan", "cover", "fuel", "intake", "turbo", "cooling", "belt", "ignition"];
const STAGGER = [...BUILD].reverse();
const SPRING = { stagger: 0.06, stiffness: 60, damping: 14, substeps: 4 };

// The camera fits every part in both poses: `--bounds` writes their corners to fit.json.
const FIT = join(HERE, "fit.json");
const fitPoints = existsSync(FIT) ? JSON.parse(readFileSync(FIT, "utf8")) : [[-560, -300, FLOOR], [460, 300, 620]];
const P = k.fitProjection(fitPoints, W, H, { pad: 18, azimuth: AZIMUTH, elevation: ELEVATION });
const V = G.viewOf(P);
const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const X = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const Z = (o) => G.frameOf(o, [0, 0, 1], [1, 0, 0], [0, 1, 0]);
const Y = (o) => G.frameOf(o, [0, 1, 0], [0, 0, 1], [1, 0, 0]);
const MOV = makeMoving(k, G, P, M);

const R = A.recorder(P, { order: STAGGER, spring: SPRING });
const { nameFor } = R;
const record = R.solid;
const taken = new Set();
function put(piece, ref, svg, { bias = 0, shapes = [], name, live, key } = {}) {
  // Every item gets a unique name: the depth-key cache is keyed by it.
  let unique = name ?? nameFor(`${piece}.item`);
  while (taken.has(unique)) unique = nameFor(name ?? `${piece}.item`);
  taken.add(unique);
  R.put({ piece, ride: piece, at: ref, bias, svg, move: PIECES[piece].move, shapes, name: unique, live, ...(key !== undefined ? { key } : {}) });
}
function box(piece, plan, z, h, style = {}, { extra = "", bias = 0, bevel = 0.6, steps = 3, name, owner, inner = "" } = {}) {
  const id = nameFor(name ?? `${piece}.box`);
  put(piece, [plan.x + plan.w / 2, plan.y + plan.d / 2, z + h / 2], S(k.slabOf(plan, z, h, P, steps, bevel), { tone: "mid", ...style, inner }) + extra, {
    bias,
    name: id,
    shapes: [record(A.box([plan.x, plan.y, z], [plan.x + plan.w, plan.y + plan.d, z + h], { name: id, owner }))],
  });
  return id;
}
function disc(piece, F, s0, s1, r, style = {}, { extra = "", bias = 0, name, owner, bevel = 0 } = {}) {
  const id = nameFor(name ?? `${piece}.disc`);
  put(piece, G.add3(F.o, G.mul3(F.a, (s0 + s1) / 2)), S(G.disc(s0, s1, r, F, P, { bevel }), { tone: "mid", ...style }) + extra, {
    bias,
    name: id,
    shapes: [record(A.disc(F, s0, s1, r, { name: id, owner }))],
  });
  return id;
}
function prism(piece, F, polygon, s0, s1, style = {}, { extra = "", bias = 0, name, owner, ref } = {}) {
  const id = nameFor(name ?? `${piece}.prism`);
  const c = polygon.reduce((acc, p) => [acc[0] + p[0] / polygon.length, acc[1] + p[1] / polygon.length], [0, 0]);
  const centre = ref ?? G.add3(G.add3(F.o, G.mul3(F.a, (s0 + s1) / 2)), G.add3(G.mul3(F.u, c[0]), G.mul3(F.v, c[1])));
  put(piece, centre, S(G.prismOf(F, polygon, s0, s1, P), { tone: "mid", ...style }) + extra, { bias, name: id, shapes: [record(A.prism(F, polygon, s0, s1, { name: id, owner }))] });
  return id;
}
const KS = G.scaleOf(P);
function tube(piece, route, r, { name, tone = "mid", maxLength = 40, bias = 0, owner, heat } = {}) {
  const id = nameFor(name ?? `${piece}.tube`);
  R.route(id, route, r, { owner: owner ?? id, piece, move: PIECES[piece].move, ride: piece });
  tubePieces(route, r, P, { maxLength, spacing: Math.max(0.8, Math.min(2.4, r * 0.18)) }).forEach((part, index) => {
    // A hot pipe: a glow along the chunk's centreline, inside its own outline, so it sorts with the pipe.
    const glow = heat ? `<path class="eng-heat" data-cyl="${heat}" d="${k.pathOf(part.points.map(at))}" style="stroke-width:${(r * 1.5 * KS).toFixed(2)}"/><path class="eng-heat eng-heat-core" data-cyl="${heat}" d="${k.pathOf(part.points.map(at))}" style="stroke-width:${(r * 0.7 * KS).toFixed(2)}"/>` : "";
    put(piece, part.mid, tubeSvg(part, { tone }) + glow, { bias, name: `${id}:${index}`, shapes: [A.tube(part.points, part.radii)] });
  });
  return id;
}

// ---------------------------------------------------------------- stand
// A shop engine stand bolted to the bellhousing face (-x): H base on casters,
// post, head tube, a turning head with four arms.
const STAND_X = -330;
const HEAD_Z = 80;
function stand() {
  const legs = [
    { x: STAND_X - 30, y: -260, w: 60, d: 520, r: 4 },
    { x: STAND_X - 180, y: -24, w: 470, d: 48, r: 4 },
  ];
  const casterAt = [[STAND_X, -235], [STAND_X, 235], [STAND_X - 160, 0], [STAND_X + 270, 0]];
  for (const [x, y] of casterAt) {
    const F = Z([x, y, FLOOR]);
    disc("stand", F, 0, 4, 22, { tone: "lo" }, { name: "stand.caster-plate" });
    disc("stand", G.frameOf([x, y - 8, FLOOR + 16], [0, 1, 0], [1, 0, 0], [0, 0, 1]), 0, 16, 16, { tone: "mid" }, { name: "stand.caster-wheel", bias: 0.5 });
  }
  box("stand", legs[1], FLOOR + 32, 36, { tone: "mid" }, { name: "stand.beam" });
  box("stand", legs[0], FLOOR + 32, 36, { tone: "mid" }, { name: "stand.cross", bias: 1, extra: D([[STAND_X, -240], [STAND_X, 240]].map(([x, y]) => at([x, y, FLOOR + 68])), { size: 0.6 }) });
  const post = { x: STAND_X - 26, y: -26, w: 52, d: 52, r: 3 };
  box("stand", post, FLOOR + 68, HEAD_Z - 30 - FLOOR - 68, { tone: "mid" }, { name: "stand.post", bias: 2, inner: L(k.sideSeam(post, FLOOR + 140, P, 3), { tone: "faint" }) });
  const sleeve = { x: STAND_X - 34, y: -34, w: 68, d: 68, r: 4 };
  box("stand", sleeve, HEAD_Z - 70, 46, { tone: "mid" }, { name: "stand.sleeve", bias: 3, extra: D([at([STAND_X - 34 + 34, 34, HEAD_Z - 47])], { size: 0.7 }) });
  // Head tube along x from the post to the turning head.
  const Ft = X([STAND_X, 0, HEAD_Z]);
  disc("stand", Ft, 0, 95, 30, { tone: "mid" }, { name: "stand.head-tube", bias: 4 });
  disc("stand", X([STAND_X + 95, 0, HEAD_Z]), 0, 14, 52, { tone: "mid" }, { name: "stand.head-flange", bias: 4.5, extra: D(G.dotsOf(X([STAND_X + 95, 0, HEAD_Z]), 14, 42, 8, P, { all: true }), { size: 0.55 }) });
  // Four arms out to the block's bellhousing bolt bosses.
  const hub = STAND_X + 109;
  for (const [y, z] of [[-58, 190], [58, 190], [-88, -40], [88, -40]]) {
    const len = Math.hypot(y, z - HEAD_Z);
    const dir = [0, y / len, (z - HEAD_Z) / len];
    const F = G.frameOf([hub, 0, HEAD_Z], dir, [1, 0, 0], G.cross3(dir, [1, 0, 0]));
    prism("stand", F, [[0, -7], [16, -7], [16, 7], [0, 7]], 20, len - 12, { tone: "mid" }, { name: "stand.arm" });
    disc("stand", X([hub, y, z]), 0, -205 - hub, 9, { tone: "mid" }, { name: "stand.arm-bolt" });
  }
}

// ---------------------------------------------------------------- block
const BLOCK = { rear: -205, front: 188, deck: M.deck, rail: -58 };
const UPPER = { x: BLOCK.rear, y: -70, w: BLOCK.front - BLOCK.rear, d: 140, r: 6 };
const LOWER = { x: BLOCK.rear, y: -98, w: BLOCK.front - BLOCK.rear, d: 196, r: 6 };
const BORE_R = M.bore / 2;
function block() {
  const cyls = [1, 2, 3, 4];
  // Bore interiors: what you see down an empty cylinder.
  const holes = cyls.map((c) => k.ring(M.cylX[c], 0, BORE_R, BLOCK.deck, P, 72));
  put("block", [0, 0, BLOCK.deck - 60], `<g class="eng-bore">${holes.map((d) => `<path d="${d}"/>`).join("")}</g>`, { name: "block.bores", bias: -400, key: -1e5 });

  // Lower crankcase, flare and the upper cylinder section with its deck.
  box("block", LOWER, BLOCK.rail, 40 - BLOCK.rail, { tone: "mid" }, {
    name: "block.crankcase",
    inner: L(k.sideSeam(LOWER, BLOCK.rail + 9, P, 3), { tone: "faint" }),
    extra:
      L([-160, -70, 20, 110].map((x) => k.sideRing([x, 98, -12], 11, "y", P, 24)).join(""), { tone: "lo" }) +
      D([-160, -70, 20, 110].map((x) => at([x, 98, -12])), { size: 0.5 }),
  });
  const Fx = X([0, 0, 0]);
  prism("block", Fx, [[-98, 40], [98, 40], [70, 120], [-70, 120]], BLOCK.rear, BLOCK.front, { tone: "mid" }, { name: "block.flare" });
  const deck = k.slabOf(UPPER, 120, BLOCK.deck - 120, P, 4, 1.2);
  deck.fill += holes.join("");
  deck.top += holes.join("");
  const ribs = [-176, -88, 0, 88].map((x) => k.segment([x, 70, 120], [x, 70, BLOCK.deck], P)).join("");
  const studs = [];
  for (const x of [-176, -88, 0, 88, 176]) for (const y of [-56, 56]) studs.push([x, y]);
  put("block", [0, 0, 170], `<g class="eng-deck">${S(deck, { tone: "mid" })}</g>` + L(ribs, { tone: "faint" }) + L(holes.join(""), { tone: "lo" }) + D(studs.map(([x, y]) => at([x, y, BLOCK.deck])), { size: 0.7, tone: "mid" }) + L(studs.map(([x, y]) => k.ring(x, y, 5.5, BLOCK.deck, P, 16)).join(""), { tone: "faint" }), {
    name: "block.upper",
    shapes: [record(A.extruded(k.ringOf(UPPER, 4), 120, BLOCK.deck - 120, { name: "block.upper" }))],
  });
  // Knock sensor and oil pressure switch bosses on the exhaust side.
  disc("block", G.frameOf([-44, 70, 160], [0, 1, 0], [1, 0, 0], [0, 0, 1]), 0, 9, 11, { tone: "mid" }, { name: "block.knock-boss", extra: L(G.circleOf(Y([-44, 70, 160]), 9, 4.5, P, 16), { tone: "lo" }) });
  disc("block", G.frameOf([44, 70, 160], [0, 1, 0], [1, 0, 0], [0, 0, 1]), 0, 9, 11, { tone: "mid" }, { name: "block.knock-boss" });
  // Front face: crank seal bore, dowels, bolt pattern for the chain cover.
  const front = [];
  for (const [y, z] of [[-60, 200], [60, 200], [-90, 20], [90, 20], [-90, -40], [90, -40], [-64, 110], [64, 110]]) front.push(at([BLOCK.front, y, z]));
  put("block", [BLOCK.front, 0, 80], D(front, { size: 0.6, tone: "mid" }) + L(k.sideRing([BLOCK.front, 0, 0], 30, "x", P, 36), { tone: "lo" }), { name: "block.front-face", bias: 0.6 });
}

// ---------------------------------------------------------------- head
const HEAD = { x: -203, y: -74, w: 393, d: 148, r: 6 };
const SPRING_DECK = 300;
const RAIL_TOP = 350;
function head() {
  const holes = [1, 2, 3, 4].map((c) => k.ring(M.cylX[c], 0, BORE_R, M.deck + 1.6, P, 72)).join("");
  box("head", HEAD, M.deck, 1.6, { tone: "lo" }, { name: "head.gasket", bevel: 0, extra: L(holes, { tone: "faint" }) });
  const studs = [];
  for (const x of [-176, -88, 0, 88, 176]) for (const y of [-46, 46]) studs.push([x, y]);
  box("head", HEAD, M.deck + 1.6, SPRING_DECK - M.deck - 1.6, { tone: "mid" }, {
    name: "head.body",
    bevel: 1,
    inner: L(k.sideSeam(HEAD, M.deck + 14, P, 6), { tone: "faint" }),
    extra:
      L([1, 2, 3, 4].flatMap((c) => [-16, 16].map((dx) => k.pathOf([[M.cylX[c] + dx - 10, 248], [M.cylX[c] + dx + 10, 248], [M.cylX[c] + dx + 10, 274], [M.cylX[c] + dx - 10, 274]].map(([x, z]) => at([x, 74, z])), true))).join(""), { tone: "lo" }) +
      D([1, 2, 3, 4].flatMap((c) => [[M.cylX[c] - 34, 240], [M.cylX[c] + 34, 282]]).map(([x, z]) => at([x, 74, z])), { size: 0.7 }) +
      D(studs.map(([x, y]) => at([x, y, SPRING_DECK])), { size: 0.8, tone: "mid" }),
  });
  // Spark plug tubes down the middle.
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    put("head", [x, 0, 320], S(k.cylinder(x, 0, 12, SPRING_DECK, RAIL_TOP - 8 - SPRING_DECK, P, 32, 0.5), { tone: "mid" }) + L(k.ring(x, 0, 7.5, RAIL_TOP - 8, P, 24), { tone: "lo" }), {
      name: "head.plug-tube",
      shapes: [record(A.cylinder(x, 0, 12, SPRING_DECK, RAIL_TOP - 8 - SPRING_DECK, { name: "head.plug-tube" }))],
    });
  }
  // Cam bearing saddles under each journal.
  for (const kind of ["intake", "exhaust"]) for (const j of MOV.MAINS) box("head", { x: j - 9, y: M.camY[kind] - 17, w: 18, d: 34, r: 1.5 }, SPRING_DECK, M.camZ - 16 - SPRING_DECK, { tone: "mid" }, { name: "head.saddle", bevel: 0.4 });
  // Rails and end walls that carry the cover.
  for (const y of [-74, 66]) box("head", { x: HEAD.x, y, w: HEAD.w, d: 8, r: 2 }, SPRING_DECK, RAIL_TOP - SPRING_DECK, { tone: "mid" }, { name: "head.rail", bevel: 0.6, extra: D([-150, -60, 30, 120].map((x) => at([x, y + 4, RAIL_TOP])), { size: 0.55 }) });
  for (const x of [HEAD.x, HEAD.x + HEAD.w - 10]) box("head", { x, y: -66, w: 10, d: 132, r: 1 }, SPRING_DECK, RAIL_TOP - SPRING_DECK, { tone: "mid" }, { name: "head.end-wall", bevel: 0.6 });
  put("head", [0, 0, 300], MOV.markup(MOV.valves(0)), { name: "head.valves", live: "valves", shapes: [record(A.box([-160, -68, 262], [160, 68, 318], { name: "head.valves" }))] });
}

function camsStatic() {
  for (const kind of ["intake", "exhaust"]) for (const j of MOV.MAINS)
    box("cams", { x: j - 9, y: M.camY[kind] - 17, w: 18, d: 34, r: 2 }, M.camZ, 14, { tone: "mid" }, { name: "cams.cap", bevel: 0.5, extra: D([at([j, M.camY[kind] - 12, M.camZ + 14]), at([j, M.camY[kind] + 12, M.camZ + 14])], { size: 0.6 }) });
  put("cams", [0, 0, M.camZ], MOV.markup(MOV.cams(0)), { name: "cams.live", live: "cams", shapes: ["intake", "exhaust"].map((kind) => record(A.disc(X([0, M.camY[kind], M.camZ]), -207, 196, 19, { name: "cams.live" }))) });
  put("cams", [208, 0, M.camZ], MOV.markup(MOV.camsFront(0)), { name: "cams.front", live: "camsFront", shapes: ["intake", "exhaust"].map((kind) => record(A.disc(X([0, M.camY[kind], M.camZ]), 196, 222, 44, { name: "cams.front" }))) });
}

// A chain loop: the hull of its sprockets (in the y–z plane at x), drawn as a link tube
// with a dashed run on top that the page slides as the crank turns.
function loopOf(x, circles, pad = 3) {
  const pts = MOV.hull(circles.flatMap(([y, z, r]) => MOV.circle(y, z, r + pad, 48)));
  return pts.map(([y, z]) => [x, y, z]);
}
function chain(piece, x, circles, driveR, name) {
  const loop = loopOf(x, circles);
  belted(piece, loop, 3, driveR, name, { tone: "mid", rings: { pitch: 6, tone: "lo" } });
  return loop;
}
// A closed loop drawn as chunks; each chunk carries its own stretch of the dashed run,
// offset by the screen length before it, so the page can slide every run as one.
function belted(piece, loop, r, driveR, name, { tone = "mid", rings = null, bias = 0 } = {}) {
  const id = nameFor(name);
  R.route(id, [...loop, loop[0]], r, { owner: id, piece, move: PIECES[piece].move, ride: piece, closed: true, limp: true });
  const scr = (p) => at(p);
  const parts = tubePieces(loop, r, P, { closed: true, maxLength: 60, spacing: 1, ...(rings ? { rings } : {}) });
  let travelled = 0;
  parts.forEach((part, index) => {
    const pts = part.points.map(scr);
    let len = 0;
    for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    const run = `<path class="eng-run" data-r="${driveR}" data-o="${travelled.toFixed(2)}" d="${k.pathOf(pts)}"/>`;
    travelled += len;
    put(piece, part.mid, tubeSvg(part, { tone }) + run, { name: `${id}:${index}`, bias, shapes: [A.tube(part.points, part.radii)] });
  });
}
function rail(piece, x, a, b, side, { name, tone = "mid", thick = 7 } = {}) {
  // A plastic rail along the run from a to b ([y, z]), on the outside of the loop (side = ±1 normal).
  const d = [b[0] - a[0], b[1] - a[1]];
  const len = Math.hypot(...d);
  const n = [(-d[1] / len) * side, (d[0] / len) * side];
  const poly = [a, b, [b[0] + n[0] * thick, b[1] + n[1] * thick], [a[0] + n[0] * thick, a[1] + n[1] * thick]].map(([y, z]) => [y + n[0] * 3.5, z + n[1] * 3.5]);
  return prism(piece, X([0, 0, 0]), poly, x - 6, x + 6, { tone }, { name });
}
function chains() {
  const S_ = MOV.SPROCKETS;
  const cam = chain("chains", S_.cam.x, [[0, 0, S_.crank[1][1]], [M.camY.intake, M.camZ, S_.cam.r], [M.camY.exhaust, M.camZ, S_.cam.r]], S_.crank[1][1], "chains.cam");
  const bal = chain("chains", S_.balance.x, [[0, 0, S_.crank[0][1]], ...S_.balance.at.map(([y, z]) => [y, z, S_.balance.r])], S_.crank[0][1], "chains.balance");
  // Guide rail on the tight (exhaust) run, tensioning rail on the slack (intake) run.
  // The straight runs are the longest hull edges on each side.
  const runOf = (loop, test) => {
    let best = null;
    loop.forEach((p, i) => {
      const q = loop[(i + 1) % loop.length];
      const len = Math.hypot(q[1] - p[1], q[2] - p[2]);
      if (test((p[1] + q[1]) / 2, (p[2] + q[2]) / 2) && (!best || len > best.len)) best = { len, a: [p[1], p[2]], b: [q[1], q[2]] };
    });
    const [lo, hi] = best.a[1] < best.b[1] ? [best.a, best.b] : [best.b, best.a];
    // Keep the rail clear of both sprockets.
    const t = (u) => [lo[0] + (hi[0] - lo[0]) * u, lo[1] + (hi[1] - lo[1]) * u];
    return [t(0.18), t(0.82)];
  };
  const [t0, t1] = runOf(cam, (y) => y > 0);
  const [s0, s1] = runOf(cam, (y) => y < 0);
  rail("chains", S_.cam.x, t0, t1, -1, { name: "chains.guide" });
  rail("chains", S_.cam.x, s0, s1, 1, { name: "chains.tension-rail" });
  // Hydraulic tensioner pushing the slack rail.
  box("chains", { x: 190, y: -92, w: 22, d: 20, r: 3 }, 90, 34, { tone: "mid" }, { name: "chains.tensioner", extra: D([at([201, -92, 124])], { size: 0.6 }) });
  const [b0, b1] = runOf(bal, (y, z) => y > 0 && z > 0);
  rail("chains", S_.balance.x, b0, b1, -1, { name: "chains.balance-guide", thick: 5 });
}

function oilpump() {
  const pump = { x: 120, y: -45, w: 86, d: 90, r: 6 };
  box("oilpump", pump, -168, 40, { tone: "mid" }, { name: "oilpump.body", bevel: 0.8, inner: L(k.sideSeam(pump, -146, P, 3), { tone: "faint" }), extra: D(k.corners(pump, 6).map(([x, y]) => at([x, y, -128])), { size: 0.6 }) });
  // Pressure stage solenoid on the side.
  disc("oilpump", Y([150, 45, -150]), 0, 14, 9, { tone: "hi" }, { name: "oilpump.solenoid", extra: L(G.circleOf(Y([150, 45, -150]), 14, 5, P, 16), { tone: "lo" }) });
  disc("oilpump", X([206, 0, MOV.SPROCKETS.pump.at[1]]), 0, 7, 8, { tone: "mid" }, { name: "oilpump.shaft" });
  const Fs = X([MOV.SPROCKETS.pump.x, 0, MOV.SPROCKETS.pump.at[1]]);
  disc("oilpump", Fs, -3.5, 3.5, MOV.SPROCKETS.pump.r, { tone: "mid" }, { name: "oilpump.sprocket", extra: D(G.dotsOf(Fs, 3.5, MOV.SPROCKETS.pump.r - 1.6, 22, P, { all: true }), { size: 0.45, tone: "lo" }) });
  chain("oilpump", MOV.SPROCKETS.pump.x, [[0, 0, MOV.SPROCKETS.crank[2][1]], [0, MOV.SPROCKETS.pump.at[1], MOV.SPROCKETS.pump.r]], MOV.SPROCKETS.crank[2][1], "oilpump.chain");
  tube("oilpump", G.fillet([[120, -20, -158], [70, -20, -158], [50, -20, -164]], 12), 7, { name: "oilpump.pickup", tone: "mid" });
  disc("oilpump", Z([38, -20, -170]), 0, 6, 22, { tone: "mid" }, { name: "oilpump.strainer", extra: D(k.dotGrid(30, -28, 4, 4, 5, -164, P), { size: 0.4, tone: "lo" }) });
}

// ---------------------------------------------------------------- covers and ancillaries
function tcover() {
  const S_ = MOV.SPROCKETS;
  const circles = [[0, 0, 30], [M.camY.intake, M.camZ, S_.cam.r + 10], [M.camY.exhaust, M.camZ, S_.cam.r + 10], ...S_.balance.at.map(([y, z]) => [y, z, S_.balance.r + 10]), [0, S_.pump.at[1], S_.pump.r + 10]];
  const outline = MOV.hull(circles.flatMap(([y, z, r]) => MOV.circle(y, z, r, 40)));
  prism("tcover", X([0, 0, 0]), outline, BLOCK.front, 232, { tone: "mid" }, { name: "tcover.cover", ref: [215, 0, 120] });
  const face = 232;
  const ring = MOV.hull(circles.flatMap(([y, z, r]) => MOV.circle(y, z, r - 9, 40)));
  const bolts = ring.filter((_, i) => i % 6 === 0).map(([y, z]) => at([face, y, z]));
  const split = 150;
  const cross = [];
  outline.forEach((p, i) => {
    const q = outline[(i + 1) % outline.length];
    if ((p[1] - split) * (q[1] - split) < 0) cross.push(p[0] + ((split - p[1]) / (q[1] - p[1])) * (q[0] - p[0]));
  });
  // Flange line, the upper/lower split, stiffening ribs that run edge to edge, bolts and the crank seal.
  const flange = k.pathOf(MOV.hull(circles.flatMap(([y, z, r]) => MOV.circle(y, z, r - 4, 40))).map(([y, z]) => at([face, y, z])), true);
  const across = (z, poly) => {
    const hits = [];
    poly.forEach((p, i) => {
      const q = poly[(i + 1) % poly.length];
      if ((p[1] - z) * (q[1] - z) < 0) hits.push(p[0] + ((z - p[1]) / (q[1] - p[1])) * (q[0] - p[0]));
    });
    return hits.length > 1 ? k.segment([face, Math.min(...hits), z], [face, Math.max(...hits), z], P) : "";
  };
  const ribs = [60, 100, 220, 260].map((z) => across(z, outline)).join("");
  put("tcover", [face, 0, 120], D(bolts, { size: 0.6 }) + L(flange, { tone: "lo" }) + L(ribs, { tone: "faint" }) + L(k.sideRing([face, 0, 0], 30, "x", P, 40), { tone: "lo" }) + L(k.sideRing([face, 0, 0], 22, "x", P, 32), { tone: "faint" }) + L(k.segment([face, Math.min(...cross), split], [face, Math.max(...cross), split], P), { tone: "lo" }), { name: "tcover.face", bias: 2 });
  // Cam adjustment valve (central valve solenoid) on the intake cam, and the crank sensor.
  const Fv = X([232, M.camY.intake, M.camZ]);
  disc("tcover", Fv, 0, 16, 21, { tone: "hi" }, { name: "tcover.cam-valve", bevel: 0.6, extra: L(G.circleOf(Fv, 16, 12, P, 32), { tone: "lo" }) + D(G.dotsOf(Fv, 16, 17, 3, P, { all: true, phase: 0.5 }), { size: 0.55 }) });
  box("tcover", { x: 236, y: M.camY.intake - 34, w: 14, d: 16, r: 2 }, M.camZ - 6, 14, { tone: "mid" }, { name: "tcover.cam-valve-plug", bevel: 0.3 });
  disc("tcover", X([232, 40, -60]), 0, 12, 10, { tone: "hi" }, { name: "tcover.crank-sensor", extra: L(G.circleOf(X([232, 40, -60]), 12, 5, P, 16), { tone: "lo" }) });
  // Top cover plug over the tensioner.
  disc("tcover", X([232, -84, 108]), 0, 6, 12, { tone: "mid" }, { name: "tcover.tensioner-plug", extra: L(G.circleOf(X([232, -84, 108]), 6, 6, P, 6), { tone: "lo" }) });
}

const PAN_UP = { x: BLOCK.rear, y: -98, w: BLOCK.front - BLOCK.rear, d: 196, r: 6 };
const SUMP = { x: -110, y: -88, w: 296, d: 176, r: 14 };
function pan() {
  box("pan", PAN_UP, -112, BLOCK.rail + 112, { tone: "mid" }, {
    name: "pan.upper",
    bevel: 1,
    inner: L(k.sideSeam(PAN_UP, BLOCK.rail - 6, P, 6), { tone: "faint" }),
    extra: L([-170, -120, -70, -20, 30, 80].map((x) => k.segment([x, 98, -112], [x, 98, BLOCK.rail - 6], P)).join(""), { tone: "faint" }) + D([-190, -140, -90, -40, 10, 60, 110, 160].map((x) => at([x, 98, BLOCK.rail - 3])), { size: 0.55 }),
  });
  box("pan", SUMP, -182, 70, { tone: "lo" }, {
    name: "pan.sump",
    bevel: 1.2,
    steps: 6,
    inner: L(k.sideSeam(SUMP, -118, P, 6), { tone: "faint" }),
    extra: D([-90, -30, 30, 90, 150].map((x) => at([x, 88, -115])), { size: 0.55 }),
  });
  disc("pan", Y([120, 88, -160]), 0, 8, 10, { tone: "hi" }, { name: "pan.drain", extra: L(G.circleOf(Y([120, 88, -160]), 8, 5, P, 6), { tone: "lo" }) });
  // Dipstick tube up the exhaust side, clipped to the block.
  const route = G.fillet([[150, 98, -40], [150, 106, -30], [150, 106, 200], [150, 96, 236]], 14);
  tube("pan", route, 4.5, { name: "pan.dipstick", tone: "hi" });
  disc("pan", Z([150, 96, 236]), 0, 8, 7, { tone: "mid" }, { name: "pan.dipstick-cap" });
  box("pan", { x: 143, y: 98, w: 14, d: 4, r: 1 }, 100, 14, { tone: "mid" }, { name: "pan.dipstick-clip", bevel: 0.3 });
}

const COVER = { x: -175, y: -74, w: 365, d: 148, r: 12 };
const COVER_TOP = 392;
function cover() {
  const coils = [1, 2, 3, 4].map((c) => k.ring(M.cylX[c], 0, 16, COVER_TOP, P, 32)).join("");
  const ribs = [-88, 0, 88].map((x) => k.lineOnTop([x, -74], [x, 74], COVER_TOP, P)).join("");
  box("cover", COVER, RAIL_TOP, COVER_TOP - RAIL_TOP, { tone: "mid" }, {
    name: "cover.body",
    bevel: 1.4,
    steps: 6,
    inner: L(k.sideSeam(COVER, RAIL_TOP + 8, P, 6), { tone: "faint" }),
    extra: L(coils, { tone: "lo" }) + L(ribs, { tone: "faint" }) + D(k.corners(COVER, 8).map(([x, y]) => at([x, y, COVER_TOP])), { size: 0.6 }) + D([-120, -30, 60, 150].flatMap((x) => [at([x, -74, RAIL_TOP + 4]), at([x, 74, RAIL_TOP + 4])]), { size: 0.5 }),
  });
  // Oil filler cap, front left.
  put("cover", [150, -42, COVER_TOP + 6], S(k.cylinder(150, -42, 22, COVER_TOP, 12, P, 40, 0.8), { tone: "hi" }) + L(k.knurl(150, -42, 22, COVER_TOP + 2, COVER_TOP + 12, 40, P), { tone: "lo" }) + S(k.slabOf({ x: 136, y: -45, w: 28, d: 6, r: 2 }, COVER_TOP + 12, 4, P, 2, 0.3), { tone: "mid" }), { name: "cover.filler", shapes: [record(A.cylinder(150, -42, 22, COVER_TOP, 12, { name: "cover.filler" }))] });
  // Oil separator (breather) at the rear with its hose barb.
  const sep = { x: -168, y: -40, w: 70, d: 70, r: 8 };
  box("cover", sep, COVER_TOP, 22, { tone: "mid" }, { name: "cover.separator", extra: L(k.planOutline(k.insetPlan(sep, 5), COVER_TOP + 22, P, 4), { tone: "faint" }) + D(k.corners(sep, 6).map(([x, y]) => at([x, y, COVER_TOP + 22])), { size: 0.55 }) });
  disc("cover", Y([-133, 30, COVER_TOP + 11]), 0, 20, 7, { tone: "hi" }, { name: "cover.breather-barb" });
  // Audi valvelift actuators over the exhaust cam: two per cylinder.
  for (const c of [1, 2, 3, 4]) for (const dx of [-26, 26]) {
    const x = M.cylX[c] + dx;
    put("cover", [x, 54, COVER_TOP + 8], S(k.cylinder(x, 54, 9, COVER_TOP, 14, P, 24, 0.5), { tone: "hi" }) + S(k.slabOf({ x: x - 4, y: 60, w: 8, d: 10, r: 1 }, COVER_TOP + 4, 7, P, 2), { tone: "mid", crease: "none" }), { name: "cover.avs", shapes: [record(A.cylinder(x, 54, 9, COVER_TOP, 14, { name: "cover.avs" }))] });
  }
  // Cam position sensors on the front.
  for (const y of [M.camY.intake, M.camY.exhaust]) disc("cover", X([190, y, COVER_TOP - 18]), 0, 14, 8, { tone: "hi" }, { name: "cover.cam-sensor" });
}

function fuel() {
  // High-pressure pump on the intake cam's four-lobe cam, rear of the head.
  const px = -201, py = M.camY.intake;
  put("fuel", [px, py, 390], S(k.cylinder(px, py, 22, 357, 20, P, 40, 0.6), { tone: "mid" }) + S(k.cylinder(px, py, 17, 377, 34, P, 36, 0.6), { tone: "hi" }) + L(k.sideArc(px, py, 17, 396, P, 24), { tone: "faint" }) + D(G.dotsOf(Z([px, py, 377]), 0, 19, 3, P, { all: true }), { size: 0.55 }), {
    name: "fuel.pump",
    shapes: [record(A.cylinder(px, py, 22, 357, 54, { name: "fuel.pump" }))],
  });
  disc("fuel", Z([px, py, 411]), 0, 7, 9, { tone: "mid" }, { name: "fuel.pump-damper" });
  // Rail along the intake side, under the runners; injectors into the head.
  const RAIL = [-150, -104, 250];
  disc("fuel", X(RAIL), 0, 300, 9, { tone: "hi" }, { name: "fuel.rail", extra: "" });
  disc("fuel", X([150, -104, 250]), 0, 16, 7, { tone: "mid" }, { name: "fuel.pressure-sensor" });
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    const from = [x, -104, 241];
    const to = [x, -76, 232];
    const dir = G.unit3(G.sub3(to, from));
    const F = G.frameAlong(from, dir);
    const len = G.len3(G.sub3(to, from));
    disc("fuel", F, 0, len, 7, { tone: "hi" }, { name: "fuel.injector" });
    disc("fuel", F, len * 0.3, len * 0.55, 9, { tone: "mid" }, { name: "fuel.injector-clip" });
    box("fuel", { x: x - 24, y: -112, w: 10, d: 16, r: 2 }, 236, 22, { tone: "mid" }, { name: "fuel.rail-mount" });
  }
  tube("fuel", G.fillet([[px, py, 411], [px, py, 424], [px, -96, 424], [-170, -104, 330], [-170, -104, 250], [-150, -104, 250]], 16), 3.2, { name: "fuel.hp-line", tone: "hi" });
}

function intake() {
  const PLENUM = { x: -175, y: -214, w: 350, d: 62, r: 22 };
  box("intake", PLENUM, 250, 82, { tone: "hi" }, { name: "intake.plenum", steps: 8, bevel: 1.6, inner: L(k.sideSeam(PLENUM, 291, P, 8), { tone: "faint" }), extra: L([-130, -45, 45, 130].map((x) => k.lineOnTop([x, -214], [x, -152], 332, P)).join(""), { tone: "faint" }) });
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    box("intake", { x: x - 30, y: -152, w: 60, d: 72, r: 8 }, 268, 34, { tone: "hi" }, { name: "intake.runner", bevel: 0.8, steps: 4 });
  }
  box("intake", { x: -175, y: -82, w: 350, d: 8, r: 2 }, 256, 52, { tone: "mid" }, { name: "intake.flange", extra: D([-150, -60, 30, 120].map((x) => at([x, -74, 300])), { size: 0.55 }) });
  // Throttle body on the front of the plenum.
  const Ft = X([175, -183, 291]);
  disc("intake", Ft, 0, 34, 30, { tone: "mid" }, { name: "intake.throttle", extra: L(G.circleOf(Ft, 34, 24, P, 40), { tone: "lo" }) + L(k.segment([209, -183 - 22, 291 - 6], [209, -183 + 22, 291 + 6], P), { tone: "mid" }) });
  box("intake", { x: 180, y: -232, w: 26, d: 22, r: 3 }, 270, 44, { tone: "mid" }, { name: "intake.throttle-motor" });
  // Runner flap actuator and its linkage at the rear.
  disc("intake", X([-205, -183, 300]), 0, 30, 16, { tone: "mid" }, { name: "intake.flap-actuator", extra: L(G.circleOf(X([-205, -183, 300]), 0, 9, P, 20), { tone: "lo" }) });
  box("intake", { x: -150, y: -200, w: 30, d: 24, r: 3 }, 332, 14, { tone: "mid" }, { name: "intake.map-sensor", extra: D([at([-135, -188, 346])], { size: 0.6 }) });
}

function turbo() {
  box("turbo", { x: -170, y: 74, w: 340, d: 10, r: 2 }, 240, 46, { tone: "mid" }, { name: "turbo.flange", extra: D([-150, -60, 30, 120].flatMap((x) => [at([x, 84, 248]), at([x, 84, 278])]), { size: 0.55 }) });
  const T = { y: 178, z: 190 };
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    tube("turbo", G.fillet([[x, 84, 263], [x, 104, 263], [x * 0.4, 132, 236], [-12, 140, 218]], 22), 14, { name: "turbo.runner", tone: "mid", maxLength: 30, heat: c });
  }
  const Fx = (x) => X([x, T.y, T.z]);
  const turbine = G.disc(0, 46, 52, Fx(-24), P);
  disc("turbo", Fx(-24), 0, 46, 52, { tone: "mid" }, { name: "turbo.turbine", extra: L(G.arcOf(Fx(-24), 23, 52, P), { tone: "faint" }) + `<path class="eng-heat-fill" data-cyl="0" d="${turbine.fill}"/>` });
  disc("turbo", Fx(-44), 0, 20, 34, { tone: "mid" }, { name: "turbo.outlet", extra: D(G.dotsOf(Fx(-44), 0, 28, 3, P, { all: true }), { size: 0.55 }) });
  disc("turbo", Fx(22), 0, 30, 25, { tone: "mid" }, { name: "turbo.chra" });
  disc("turbo", Fx(52), 0, 40, 58, { tone: "hi" }, { name: "turbo.compressor", bevel: 0.8, extra: L(G.arcOf(Fx(52), 20, 58, P), { tone: "faint" }) });
  disc("turbo", Fx(92), 0, 18, 36, { tone: "hi" }, { name: "turbo.inlet", extra: L(G.circleOf(Fx(92), 18, 28, P, 48), { tone: "lo" }) + L(G.spokesOf(Fx(92), 18, 8, 28, 7, P, { sweep: 0.5 }), { tone: "faint" }) + L(G.circleOf(Fx(92), 18, 8, P, 20), { tone: "lo" }) });
  // Compressor outlet toward the charge pipe.
  const Fo = Z([72, T.y, T.z + 50]);
  disc("turbo", Fo, 0, 40, 18, { tone: "hi" }, { name: "turbo.charge-out" });
  // Wastegate actuator beside the compressor, rod back to the turbine.
  disc("turbo", X([60, 240, 150]), 0, 44, 22, { tone: "mid" }, { name: "turbo.wastegate", extra: L(G.circleOf(X([60, 240, 150]), 44, 12, P, 24), { tone: "lo" }) });
  box("turbo", { x: 40, y: 214, w: 30, d: 30, r: 3 }, 142, 8, { tone: "mid" }, { name: "turbo.wg-bracket" });
  disc("turbo", X([-8, 240, 150]), 0, 68, 3, { tone: "hi" }, { name: "turbo.wg-rod" });
  // Oil feed and return, coolant feed and return, to the block's exhaust side.
  tube("turbo", G.fillet([[30, 178, 215], [30, 178, 236], [30, 150, 236], [100, 120, 160], [100, 104, 20], [100, 98, 20]], 16), 3.2, { name: "turbo.oil-feed", tone: "hi" });
  tube("turbo", G.fillet([[34, 178, 165], [34, 178, 120], [34, 120, 60], [34, 104, -20], [34, 98, -20]], 18), 6, { name: "turbo.oil-return", tone: "mid" });
  tube("turbo", G.fillet([[14, 203, 190], [-6, 230, 190], [-6, 230, 70], [-70, 104, 0], [-70, 98, 0]], 18), 4, { name: "turbo.coolant", tone: "mid" });
}

function cooling() {
  // Coolant module on the intake side of the crankcase.
  box("cooling", { x: -60, y: -128, w: 90, d: 30, r: 6 }, -40, 60, { tone: "hi" }, { name: "cooling.module", bevel: 1, extra: D([at([-50, -128, 12]), at([20, -128, 12]), at([-50, -128, -32]), at([20, -128, -32])], { size: 0.55 }) });
  disc("cooling", Y([-15, -128, -10]), 0, -26, 18, { tone: "mid" }, { name: "cooling.thermostat" });
  tube("cooling", G.fillet([[-15, -154, -10], [-15, -170, -10], [-15, -170, 60], [-90, -170, 60]], 18), 13, { name: "cooling.hose-upper", tone: "mid" });
  tube("cooling", G.fillet([[20, -128, -30], [70, -128, -30], [100, -112, -30]], 14), 8, { name: "cooling.pipe", tone: "mid" });
  // Oil filter housing and cartridge cap, front of the intake side.
  box("cooling", { x: 120, y: -112, w: 40, d: 14, r: 3 }, 30, 60, { tone: "mid" }, { name: "cooling.filter-bracket" });
  put("cooling", [140, -140, 65], S(k.cylinder(140, -140, 34, 22, 84, P, 56, 1), { tone: "hi" }) + L(k.sideArc(140, -140, 34, 92, P, 40), { tone: "faint" }) + S(k.cylinder(140, -140, 22, 106, 10, P, 6, 0.6), { tone: "mid" }), {
    name: "cooling.filter",
    shapes: [record(A.cylinder(140, -140, 34, 22, 94, { name: "cooling.filter" }))],
  });
}

function belt() {
  const plane = 250;
  const pulleys = [[0, 0, 76], [-150, 180, 28], [-160, -60, 55], [-70, 150, 24]];
  // Vibration damper on the crank nose.
  const Fd = X([236, 0, 0]);
  disc("belt", Fd, 0, 26, 76, { tone: "hi" }, { name: "belt.damper", bevel: 0.8, extra: L(G.circleOf(Fd, 26, 40, P, 64), { tone: "lo" }) + D(G.dotsOf(Fd, 26, 22, 4, P, { all: true }), { size: 0.6 }) + L(G.ribsOf(Fd, 0, 26, 76, 96, P), { tone: "faint" }) });
  // Alternator, A/C compressor and tensioner, each with its pulley in the belt plane.
  disc("belt", X([110, -150, 180]), 0, 128, 58, { tone: "mid" }, { name: "belt.alternator", extra: L([30, 64, 98].map((s) => G.arcOf(X([110, -150, 180]), s, 58, P)).join(""), { tone: "faint" }) });
  disc("belt", X([238, -150, 180]), 0, 24, 28, { tone: "hi" }, { name: "belt.alt-pulley", extra: L(G.circleOf(X([238, -150, 180]), 24, 9, P, 20), { tone: "lo" }) });
  disc("belt", X([100, -160, -60]), 0, 132, 55, { tone: "mid" }, { name: "belt.ac", extra: L([40, 90].map((s) => G.arcOf(X([100, -160, -60]), s, 55, P)).join(""), { tone: "faint" }) });
  disc("belt", X([232, -160, -60]), 0, 30, 55, { tone: "hi" }, { name: "belt.ac-clutch", extra: L(G.circleOf(X([232, -160, -60]), 30, 20, P, 40), { tone: "lo" }) + D(G.dotsOf(X([232, -160, -60]), 30, 12, 3, P, { all: true }), { size: 0.55 }) });
  box("belt", { x: 214, y: -86, w: 24, d: 30, r: 4 }, 128, 40, { tone: "mid" }, { name: "belt.tensioner-arm" });
  disc("belt", X([238, -70, 150]), 0, 24, 24, { tone: "hi" }, { name: "belt.tensioner", extra: L(G.circleOf(X([238, -70, 150]), 24, 8, P, 20), { tone: "lo" }) });
  belted("belt", loopOf(plane, pulleys, 2), 3, 76, "belt.belt", { tone: "mid", bias: 3 });
}

function ignition() {
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    put("ignition", [x, 0, COVER_TOP + 20], S(k.cylinder(x, 0, 14, COVER_TOP, 18, P, 32, 0.5), { tone: "mid" }) + S(k.slabOf({ x: x - 18, y: -22, w: 36, d: 44, r: 5 }, COVER_TOP + 18, 16, P, 4, 0.8), { tone: "hi" }) + S(k.slabOf({ x: x - 8, y: 22, w: 16, d: 12, r: 2 }, COVER_TOP + 22, 10, P, 2, 0.4), { tone: "mid" }), {
      name: "ignition.coil",
      shapes: [record(A.box([x - 18, -22, COVER_TOP], [x + 18, 22, COVER_TOP + 34], { name: "ignition.coil" }))],
    });
  }
  // Harness along the exhaust side of the coils, one branch to each connector.
  const trunk = G.fillet([[-230, 44, COVER_TOP + 27], [-160, 44, COVER_TOP + 27], [170, 44, COVER_TOP + 27], [205, 44, COVER_TOP + 10], [205, 44, COVER_TOP - 18]], 18);
  tube("ignition", trunk, 5, { name: "ignition.harness", tone: "lo", maxLength: 40 });
  for (const c of [1, 2, 3, 4]) {
    const x = M.cylX[c];
    tube("ignition", [[x, 34, COVER_TOP + 27], [x, 39, COVER_TOP + 27]], 2.6, { name: "ignition.branch", tone: "lo" });
  }
  disc("ignition", X([-230, 44, COVER_TOP + 27]), -12, 0, 9, { tone: "mid" }, { name: "ignition.connector" });
}

// ---------------------------------------------------------------- moving parts
put("crank", [0, 0, 0], MOV.markup(MOV.crank(0)), {
  name: "crank.live",
  live: "crank",
  shapes: [record(A.disc(X([0, 0, 0]), -200, 187, 74, { name: "crank.live" }))],
});
put("crank", [214, 0, 0], MOV.markup(MOV.crankFront(0)), {
  name: "crank.front",
  live: "crankFront",
  shapes: [record(A.disc(X([0, 0, 0]), 187, 241, 24, { name: "crank.front" }))],
});
put("pistons", [0, 0, 120], MOV.markup(MOV.pistons(0)), {
  name: "pistons.live",
  live: "pistons",
  // Pistons at rest, and the volume each rod sweeps down to the crank pin.
  shapes: [1, 2, 3, 4].flatMap((c) => [
    record(A.cylinder(M.cylX[c], 0, BORE_R - 0.25, M.pistonPin(0, c) - M.skirt, M.skirt + M.compressionHeight, { name: "pistons.live" })),
    record(A.box([M.cylX[c] - 10, -60, -M.r - 31], [M.cylX[c] + 10, 60, M.pistonPin(0, c) - M.skirt], { name: "pistons.live" })),
  ]),
});

put("balance", [0, 0, 0], MOV.markup(MOV.balance(0)), { name: "balance.live", live: "balance", shapes: MOV.SPROCKETS.balance.at.map(([y, z]) => record(A.disc(X([0, y, z]), -190, 188, 30, { name: "balance.live" }))) });
put("balance", [196, 0, 0], MOV.markup(MOV.balanceFront(0)), { name: "balance.front", live: "balanceFront", shapes: MOV.SPROCKETS.balance.at.map(([y, z]) => record(A.disc(X([0, y, z]), 188, 201, 24, { name: "balance.front" }))) });

stand();
block();
head();
camsStatic();
chains();
oilpump();
tcover();
pan();
cover();
fuel();
intake();
turbo();
cooling();
belt();
ignition();

// ---------------------------------------------------------------- page
if (process.argv.includes("--bounds")) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const item of R.items) for (const shape of item.shapes ?? []) {
    const b = A.boundsOf(shape);
    for (const share of [0, 1]) for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i], b.min[i] + item.move[i] * share);
      hi[i] = Math.max(hi[i], b.max[i] + item.move[i] * share);
    }
  }
  console.log("bounds", lo.map(Math.round), hi.map(Math.round));
  const corners = [];
  for (const item of R.items) for (const shape of item.shapes ?? []) {
    const b = A.boundsOf(shape);
    for (const share of [0, 1]) for (const x of [b.min[0], b.max[0]]) for (const y of [b.min[1], b.max[1]]) for (const z of [b.min[2], b.max[2]]) corners.push([x + item.move[0] * share, y + item.move[1] * share, z + item.move[2] * share].map((v) => Math.round(v)));
  }
  const seen = new Set();
  writeFileSync(FIT, JSON.stringify(corners.filter((c) => { const key = c.join(); if (seen.has(key)) return false; seen.add(key); return true; })));
}
// Depth keys: settling them takes minutes, so a run caches them and `--cached` reuses them.
const KEYS = join(HERE, "keys.json");
if (process.argv.includes("--cached") && existsSync(KEYS)) {
  const cache = JSON.parse(readFileSync(KEYS, "utf8"));
  let missing = 0;
  for (const item of R.items) if (item.name in cache) item.key = cache[item.name]; else missing++;
  if (missing) console.log(`keys.json: ${missing} items not in the cache`);
} else if (!process.argv.includes("--fast")) {
  const t0 = Date.now();
  // The page shows build-step states: assemblies go on in order, one moving at a time.
  const states = [0, 1, ...A.prefixStates(STAGGER, { shares: [0.5, 1] })];
  A.settle(R, P, { states, step: 0.4 });
  writeFileSync(KEYS, JSON.stringify(Object.fromEntries(R.items.map((item) => [item.name, Number(item.key.toFixed(3))]))));
  console.log(`settled ${R.items.length} items in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
R.items.sort((a, b) => a.key - b.key);
const groups = R.items.map((item, index) => {
  const [mx, my, mz] = item.move;
  return `<g class="it" data-piece="${item.piece}" data-k="${item.key.toFixed(2)}" data-m="${mx.toFixed(1)},${my.toFixed(1)},${mz.toFixed(1)}" data-i="${index}"${item.live ? ` data-live="${item.live}"` : ""}>${item.svg}</g>`;
});

const LABEL =
  "An Audi EA888 Gen 2 2.0 TFSI engine, code CAEB, built up on a shop engine stand. The cast-iron block is bolted by its bellhousing face to the stand's turning head; the crankshaft, pistons and rods and the rest of the engine wait apart around it and go on in build order, while the crank turns and the pistons move with it.";
const svg = `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${LABEL}"><defs><filter id="eng-blur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="2.4"/></filter><filter id="eng-blur-wide" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="5"/></filter></defs><g class="eng-engine">${groups.join("")}</g></svg>`;

const DATA = { V, P, scale: G.scaleOf(P), pieces: Object.fromEntries(Object.entries(PIECES).map(([id, p]) => [id, { name: p.name, say: p.say, move: p.move }])), build: BUILD, spring: SPRING };
const latheSrc = readFileSync(join(KIT, "lathe.mjs"), "utf8").replace(/^import .*$/m, "").replace(/^export /gm, "");
const latheNames = [...latheSrc.matchAll(/^(?:function|const) ([A-Za-z0-9_]+)/gm)].map((m) => m[1]);
const runtime = `
const G = (() => { ${latheSrc}
return { ${latheNames.join(", ")} }; })();
const makeModel = ${makeModel.toString()};
const makeMoving = ${makeMoving.toString()};
`;
const LIVE = readFileSync(join(HERE, "live.js"), "utf8").replace("__DATA__", JSON.stringify(DATA));

const steps = BUILD.map((id, index) => `<button type="button" class="eng-step" data-step="${index + 1}"><span class="eng-num">${String(index + 1).padStart(2, "0")}</span>${PIECES[id].name}</button>`).join("");
const corners = `<div class="iso-plate-corners" aria-hidden="true"><span class="iso-plate-corner" data-corner="fig">Fig 1</span><span class="iso-plate-corner" data-corner="title">EA888 Gen 2 · CAEB</span><span class="iso-plate-corner" data-corner="hint">Hover or ← → to build · drag to turn the crank</span><span class="iso-plate-corner" data-corner="readout" data-readout>bare block</span></div>`;
const controls = `<div class="eng-controls"><button type="button" class="eng-start" data-start disabled><span class="eng-lamp" aria-hidden="true"></span><span data-start-label>Start</span></button><span class="eng-note" data-note>Finish the build to start the engine</span></div>`;
const stage = `<div class="eng-stage" tabindex="0" role="slider" aria-label="EA888 engine build. Arrow keys add or remove assemblies in build order; Space turns the crank." aria-valuemin="0" aria-valuemax="${BUILD.length}" aria-valuenow="0">${svg}</div>`;
const caption =
  "The 2011 Audi A4's 2.0 TFSI (engine code CAEB) is a second-generation EA888: a turbocharged, direct-injected inline four of 1,984 cc, 82.5 mm bore by 92.8 mm stroke on 144 mm rods, 9.6:1 compression, 155 kW (211 hp) and 350 Nm, firing 1-3-4-2. Illustration, not service data: proportions are true where noted, and torque specs and clearances belong in the factory repair manual.";
const body = `<figure class="iso-figure eng"><div class="iso-plate">${existsSync(join(HERE, "ea888-caeb-detailed-a4.pdf")) ? pdfButton("ea888-caeb-detailed-a4.pdf") : ""}${corners}${stage}${controls}</div><figcaption class="iso-legend"><span class="eng-steps">${steps}</span><span class="iso-caption">${caption}</span></figcaption></figure>`;
const CSS = readFileSync(join(HERE, "engine.css"), "utf8");
const page = k.pageHtml({ title: "EA888 Gen 2 engine build", theme: THEME, body: `<style>${CSS}</style>${body}`, script: kitScript() + runtime + LIVE, width: 1040 });
const out = THEME === "light" ? "ea888-caeb-light.html" : "ea888-caeb.html";
writeFileSync(join(HERE, out), page);
console.log("wrote", out, `${(page.length / 1024).toFixed(0)} KB`, `${R.items.length} items`);

// Audit the poses the page shows: assembled, apart, and every build step half-way and done.
A.auditOrExit(R, P, { frames: [0, 1, ...A.prefixStates(STAGGER, { shares: [0.5, 1] })] });
