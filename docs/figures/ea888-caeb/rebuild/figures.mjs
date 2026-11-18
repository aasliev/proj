// Close-up figures for the rebuild trainer. Each figure is one SVG; every part sits in
// <g data-part="id" data-apart="dx,dy"> so the page can hold it apart (shifted by the
// screen offset) until its step is done, then slide it home.
import * as k from "../../../../.claude/skills/anatomy/kit/iso-kit.mjs";
import * as G from "../../../../.claude/skills/anatomy/kit/lathe.mjs";
import { tubePieces, tubeSvg } from "../../../../.claude/skills/anatomy/kit/tube.mjs";

const DEG = Math.PI / 180;
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const frame = (o, a, u, v) => G.frameOf(o, a, u, v);
const X = (o) => frame(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
const Y = (o) => frame(o, [0, 1, 0], [0, 0, 1], [1, 0, 0]);
const Z = (o) => frame(o, [0, 0, 1], [1, 0, 0], [0, 1, 0]);

// A figure: fit a camera, then collect parts in painter's order.
function figure({ id, width = 560, height = 400, fit, azimuth = 40, elevation = 28, pad = 26, label }) {
  const P = k.fitProjection(fit, width, height, { pad, azimuth, elevation });
  const V = G.viewOf(P);
  const at = (p) => k.iso(p, P);
  const parts = [];
  const api = {
    P,
    V,
    at,
    depth: (p) => G.dot3(p, V),
    // A part: its markup, and where it waits before its step (a world offset).
    part(name, svg, apart = [0, 0, 0]) {
      const o = at([0, 0, 0]);
      const m = at(apart);
      parts.push(`<g data-part="${name}" data-apart="${(m[0] - o[0]).toFixed(1)},${(m[1] - o[1]).toFixed(1)}">${svg}</g>`);
    },
    // Several solids sorted far to near.
    sorted(list) {
      return list.sort((a, b) => a[0] - b[0]).map((entry) => entry[1]).join("");
    },
    svg() {
      return `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg rb-fig" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}" data-fig="${id}">${parts.join("")}</svg>`;
    },
  };
  return api;
}

// A curved bar of an annulus (shell, saddle, ring) along an axis frame: convex segments
// filled without seams, then one set of edges. Angles in degrees from F.u toward F.v.
// Returns [[depth, svg]] for sorting with neighbours.
function arcSolids(F, r0, r1, a0, a1, s0, s1, P, V, { tone = "hi", n = 0, lit = false, edge = "mid" } = {}) {
  const count = n || Math.max(4, Math.ceil(Math.abs(a1 - a0) / 10));
  const segs = [];
  for (let i = 0; i < count; i++) {
    const t0 = (a0 + ((a1 - a0) * i) / count) * DEG;
    const t1 = (a0 + ((a1 - a0) * (i + 1)) / count) * DEG;
    const poly = [
      [r0 * Math.cos(t0), r0 * Math.sin(t0)],
      [r1 * Math.cos(t0), r1 * Math.sin(t0)],
      [r1 * Math.cos(t1), r1 * Math.sin(t1)],
      [r0 * Math.cos(t1), r0 * Math.sin(t1)],
    ];
    const paths = G.prismOf(F, poly, s0, s1, P);
    paths.outline = "";
    paths.crease = "";
    paths.seamless = true;
    const mid = G.pointOf(F, (s0 + s1) / 2, (r0 + r1) / 2, (t0 + t1) / 2);
    segs.push([G.dot3(mid, V), S(paths, { tone, crease: "none", lit })]);
  }
  segs.sort((p, q) => p[0] - q[0]);
  // Edges: rims of both faces (full on the face toward the viewer), the visible part of
  // the outer and inner surfaces' rims on the far face, and the two cut ends.
  const near = G.dot3(F.a, V) > 0 ? s1 : s0;
  const far = near === s1 ? s0 : s1;
  const from = Math.min(a0, a1) * DEG, to = Math.max(a0, a1) * DEG;
  const rim = [
    G.arcOf(F, near, r1, P, { from, to, least: -1 }),
    G.arcOf(F, near, r0, P, { from, to, least: -1 }),
    G.arcOf(F, far, r1, P, { from, to }),
    G.arcOf(F, far, r0, P, { from, to, inward: true }),
  ];
  const ends = [a0, a1].map((a) => {
    const t = a * DEG;
    const q = (s, r) => k.iso(G.pointOf(F, s, r, t), P);
    return k.pathOf([q(far, r0), q(near, r0), q(near, r1), q(far, r1)]);
  });
  const lines = L(rim.join("") + ends.join(""), { tone: edge, className: "rb-edge" });
  const mid = G.pointOf(F, (s0 + s1) / 2, (r0 + r1) / 2, ((a0 + a1) / 2) * DEG);
  return [[G.dot3(mid, V), segs.map((x) => x[1]).join("") + lines]];
}

// ---------------------------------------------------------------- main bearing station
// One bearing station drawn apart vertically, block upside down as it is on the stand:
// saddle and bulkhead at the bottom, then the upper shell, the journal, the lower shell,
// the cap and its bolts on top. Crank axis along x.
function mains() {
  const OFF = { shellUp: 70, thrust: 125, crank: 215, shellLow: 305, caps: 360, bolts: 450 };
  const f = figure({ id: "mains", label: "One main bearing station drawn apart: block saddle, upper shell with oil groove, thrust washers, crankshaft journal, lower shell, bearing cap, two cap bolts and a side bolt.", fit: [[-30, -72, -64], [30, 72, -64], [-30, -72, 545], [30, 72, 545]], azimuth: 32, elevation: 22, height: 480 });
  const { P, V, at } = f;
  const R = 29; // journal radius: 58 mm pin
  const W = 13; // half width of the bearing
  const F = X([0, 0, 0]); // crank axis; u = +y, v = +z
  const HOUSE = 62;
  // Block bulkhead, upside down on the stand: its saddle opens upward.
  const boss = (y, z0, z1, tone = "mid") => S(k.cylinder(0, y, 9, z0, z1 - z0, P, 24, 0.4), { tone });
  const block = arcSolids(F, R + 2.2, HOUSE, 180, 360, -W - 4, W + 4, P, V, { tone: "mid" });
  f.part("block", block[0][1] + L([-48, 48].map((y) => k.ring(0, y, 4.5, 0, P, 16)).join(""), { tone: "lo" }));
  // Upper shell (in the block): oil groove and locating tab.
  const up = arcSolids(F, R, R + 2.2, 180, 360, -W, W, P, V, { tone: "hi" });
  const groove = G.arcOf(F, 0, R + 0.05, P, { from: 200 * DEG, to: 340 * DEG, inward: true });
  f.part("shellUp", up[0][1] + L(groove, { tone: "lo" }) + S(k.slabOf({ x: W - 5, y: -R - 3, w: 4, d: 3, r: 0.5 }, -1.5, 1.5, P, 2), { tone: "hi" }), [0, 0, OFF.shellUp]);
  // Thrust washers (bearing 3 only): half rings against both faces of the bearing.
  const tw = [...arcSolids(F, R + 2, R + 13, 180, 360, -W - 2.4, -W, P, V, { tone: "hi" }), ...arcSolids(F, R + 2, R + 13, 180, 360, W, W + 2.4, P, V, { tone: "hi" })];
  f.part("thrust", f.sorted(tw), [0, 0, OFF.thrust]);
  // A slice of the crankshaft: journal with a web each side, crank pin pointing up.
  const crank = [[G.dot3([0, 0, 0], V), S(G.disc(-W - 0.5, W + 0.5, R, F, P, { steps: 64 }), { tone: "hi" }) + L(G.arcOf(F, 0, R, P), { tone: "faint" })]];
  for (const [s0, s1] of [[-W - 17, -W - 0.5], [W + 0.5, W + 17]]) crank.push([G.dot3([(s0 + s1) / 2, 0, 0], V), S(G.prismOf(F, web(), s0, s1, P), { tone: "mid" })]);
  f.part("crank", f.sorted(crank), [0, 0, OFF.crank]);
  // Lower shell (in the cap): no groove.
  const low = arcSolids(F, R, R + 2.2, 0, 180, -W, W, P, V, { tone: "hi" });
  f.part("shellLow", low[0][1], [0, 0, OFF.shellLow]);
  // Bearing cap: the other half of the housing, with its bolt bosses and the side bolt hole.
  const cap = arcSolids(F, R + 2.2, HOUSE, 0, 180, -W - 4, W + 4, P, V, { tone: "mid" });
  f.part("caps", cap[0][1] + boss(-48, 0, 40) + boss(48, 0, 40) + L(k.ring(0, -48, 4.5, 40, P, 16) + k.ring(0, 48, 4.5, 40, P, 16), { tone: "lo" }), [0, 0, OFF.caps]);
  // Two cap bolts and a side bolt into the cap's side.
  const bolt = (y) => S(k.cylinder(0, y, 4, 40, 34, P, 16), { tone: "mid", crease: "none" }) + S(k.cylinder(0, y, 7, 74, 6, P, 24, 0.4), { tone: "hi" });
  const SB = Y([0, HOUSE - 4, 24]);
  const side = S(G.disc(0, 30, 3.5, SB, P), { tone: "mid" }) + S(G.disc(30, 36, 6.5, SB, P), { tone: "hi" });
  f.part("bolts", bolt(-48) + bolt(48) + side, [0, 0, OFF.bolts]);
  return f.svg();
}
// A crank web in the y–z plane: hub round the journal, the pin above, a round counterweight below.
function web() {
  const pts = [];
  for (let t = 0; t < 360; t += 15) pts.push([33 * Math.cos(t * DEG), 33 * Math.sin(t * DEG)]);
  for (let t = 0; t < 360; t += 20) pts.push([25 * Math.cos(t * DEG), 46.4 + 25 * Math.sin(t * DEG)]);
  for (let t = -75; t <= 75; t += 10) pts.push([60 * Math.sin(t * DEG), -60 * Math.cos(t * DEG)]);
  return hull(pts);
}
function hull(points) {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], hi = [];
  for (const p of pts) { while (lo.length > 1 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of pts.reverse()) { while (hi.length > 1 && cross(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); }
  return lo.slice(0, -1).concat(hi.slice(0, -1));
}

// ---------------------------------------------------------------- piston, pin, clips and rod
const BORE = 41.25;
function pistonBody(f, z0, { lit = false } = {}) {
  const { P } = f;
  const r = BORE - 0.4;
  const top = z0 + 56.6;
  const rings = [4, 7.5, 11].map((d) => k.sideArc(0, 0, r, top - d, P, 48)).join("");
  // Arrow on the crown, toward the belt pulley end (+x here): a shaft and a head.
  const arrow = k.lineOnTop([-14, 0], [14, 0], top, P) + k.lineOnTop([14, 0], [7, -5], top, P) + k.lineOnTop([14, 0], [7, 5], top, P);
  const reliefs = [[-12, -16], [-12, 16], [12, -16], [12, 16]].map(([x, y]) => k.ring(x, y, 9, top, P, 24)).join("");
  return S(k.cylinder(0, 0, r, z0, 56.6, P, 72, 0.8), { tone: "hi", crease: "lo", lit, inner: L(rings, { tone: "lo" }) }) + L(reliefs, { tone: "faint" }) + L(arrow, { tone: "hi", className: "rb-arrow" }) + L(k.sideRing([0, r, z0 + 26], 10, "y", P, 28), { tone: "lo" });
}
function rodSvg(f, z, pinZ, { capOnly = false, rodOnly = false } = {}) {
  const { P } = f;
  // Rod in the x–z plane, 20 thick along y; big end at z, small end at pinZ.
  const F = Y([0, -10, 0]);
  const big = [], small = [];
  for (let t = 0; t < 360; t += 15) big.push([31 * Math.cos(t * DEG), z + 31 * Math.sin(t * DEG)]);
  for (let t = 0; t < 360; t += 20) small.push([13 * Math.cos(t * DEG), pinZ + 13 * Math.sin(t * DEG)]);
  // u = z, v = x for frame Y; build polygons as [z, x].
  const flip = (pts) => pts.map(([x, zz]) => [zz, x]);
  const beam = hull([...big.filter(([, zz]) => zz >= z), ...small]);
  const cap = hull([...big.filter(([, zz]) => zz <= z), [-31, z], [31, z]]);
  const out = [];
  if (!capOnly) out.push(S(G.prismOf(F, flip(beam), 0, 20, P), { tone: "mid" }) + L(G.circleOf(Y([0, 10, pinZ]), 0, 10.5, P, 24), { tone: "lo" }) + D([k.iso([20, 10, z + 12], P)], { size: 0.9, tone: "hi" }));
  if (!rodOnly) out.push(S(G.prismOf(F, flip(cap), 0, 20, P), { tone: "mid" }) + D([k.iso([20, 10, z - 12], P)], { size: 0.9, tone: "hi" }));
  return out.join("");
}
function piston() {
  const f = figure({ id: "piston", label: "A piston with its pin and two circlips drawn out to the side, and the connecting rod below. The arrow on the crown points to the belt pulley end.", fit: [[-60, -110, -150], [60, 110, -150], [-60, -110, 70], [60, 110, 70]], azimuth: 30, elevation: 26 });
  const { P } = f;
  f.part("rod", rodSvg(f, -120, 0), [0, 0, -40]);
  // Pin and circlips sit inside the piston once fitted, so the piston is drawn over them.
  const pin = Y([0, -BORE, 0]);
  f.part("ppin", S(G.disc(4, 2 * BORE - 4, 10, pin, P), { tone: "hi" }) + L(G.circleOf(pin, 2 * BORE - 4, 6, P, 20), { tone: "lo" }), [0, 110, 0]);
  const clip = (y) => S(G.ringBand(-0.8, 0.8, 8, 11, Y([0, y, 0]), P), { tone: "mid" });
  f.part("clips", clip(-BORE + 2) + clip(BORE - 2), [0, 140, 0]);
  f.part("piston", pistonBody(f, -26));
  return f.svg();
}

// ---------------------------------------------------------------- piston rings
// A ring as a round wire with its gap; `half` keeps only the far (-1) or near (+1) side of the piston.
function ringTube(f, z, r, gapDeg, rr, tone, half) {
  const pts = [];
  for (let t = gapDeg + 6; t <= gapDeg + 354; t += 3) {
    const p = [r * Math.cos(t * DEG), r * Math.sin(t * DEG), z];
    const near = p[0] * f.V[0] + p[1] * f.V[1] > 0;
    pts.push(near === (half > 0) ? p : null);
  }
  const runs = [];
  let run = [];
  for (const p of pts) { if (p) run.push(p); else { if (run.length > 1) runs.push(run); run = []; } }
  if (run.length > 1) runs.push(run);
  return runs.map((points) => tubePieces(points, rr, f.P, { maxLength: 400, spacing: 1 }).map((piece) => tubeSvg(piece, { tone })).join("")).join("");
}
function rings() {
  const f = figure({ id: "rings", label: "A piston with its three rings: top compression ring, second compression ring and the oil scraper ring, their gaps turned 120 degrees apart.", fit: [[-50, -50, -30], [50, 50, -30], [-50, -50, 120], [50, 50, 120]], azimuth: 30, elevation: 30 });
  const top = 30.6;
  const set = (half) => ringTube(f, top - 3, BORE + 0.6, 0, 1.4, "hi", half) + ringTube(f, top - 7, BORE + 0.6, 120, 1.4, "hi", half) + ringTube(f, top - 11, BORE + 0.6, 240, 1.9, "mid", half);
  f.part("ringsBack", set(-1), [0, 0, 70]);
  f.part("piston", pistonBody(f, -26));
  f.part("rings", set(1), [0, 0, 70]);
  return f.svg();
}

// ---------------------------------------------------------------- rod, cap and shells
function rod() {
  const f = figure({ id: "rod", label: "A connecting rod with its cap drawn apart, the two bearing shells centred between them and the two new rod bolts.", fit: [[-60, -30, -230], [60, 30, -230], [-60, -30, 40], [60, 30, 40]], azimuth: 30, elevation: 20 });
  const { P, V } = f;
  const z = -120;
  f.part("rod", rodSvg(f, z, 0, { rodOnly: true }));
  const F = Y([0, -10, z]);
  const shellUp = arcSolids(Y([0, 0, z]), 23.9, 26, 0, 180, -9, 9, P, V, { tone: "hi" });
  const shellLow = arcSolids(Y([0, 0, z]), 23.9, 26, 180, 360, -9, 9, P, V, { tone: "hi" });
  void F;
  f.part("rodShells", shellUp[0][1] + `<g transform="translate(0 0)">${shellLow[0][1]}</g>`, [0, 0, -30]);
  f.part("rodCap", rodSvg(f, z, 0, { capOnly: true }), [0, 0, -60]);
  const bolt = (x) => S(k.cylinder(x, 0, 3.5, z - 70, 40, P, 16), { tone: "mid", crease: "none" }) + S(k.cylinder(x, 0, 6.5, z - 76, 6, P, 24, 0.4), { tone: "hi" });
  f.part("rodBolts", bolt(-22) + bolt(22), [0, 0, -90]);
  return f.svg();
}

// ---------------------------------------------------------------- balance shafts
function balanceShaft(f, y, z, turn) {
  const { P, V } = f;
  const F = X([0, y, z]);
  const list = [];
  const segs = [[-190, -150, 10, "mid"], [-150, -130, 15, "hi"], [-40, -10, 10, "mid"], [-10, 10, 15, "hi"], [10, 40, 10, "mid"], [130, 150, 15, "hi"], [150, 190, 10, "mid"]];
  for (const [a, b, r, tone] of segs) list.push([G.dot3([(a + b) / 2, y, z], V), S(G.disc(a, b, r, F, P), { tone })]);
  for (const [a, b] of [[-130, -40], [40, 130]]) {
    const half = [[0, 0]];
    for (let t = -90; t <= 90; t += 10) half.push([30 * Math.sin((turn + t) * DEG), 30 * Math.cos((turn + t) * DEG)]);
    list.push([G.dot3([(a + b) / 2, y, z], V), S(G.disc(a, b, 10, F, P), { tone: "hi" }) + S(G.prismOf(F, hull(half), a, b, P), { tone: "mid" })]);
  }
  list.push([G.dot3([195, y, z], V), S(G.disc(190, 200, 22, F, P), { tone: "mid" }) + D(G.dotsOf(F, 200, 20, 20, P, { all: true }), { size: 0.45, tone: "lo" })]);
  return f.sorted(list);
}
function balance() {
  const f = figure({ id: "balance", label: "The two balance shafts drawn out of the front of the block, the exhaust side shaft's pipe, the intermediate sprocket on its mounting pin, and the intake side seal.", fit: [[-200, -120, -80], [540, -120, -80], [-200, 120, 80], [540, 120, 80]], azimuth: 34, elevation: 26 });
  const { P } = f;
  // Block front face as context: a plate with the two shaft bores.
  const plate = { x: 180, y: -110, w: 8, d: 220, r: 4 };
  f.part("pipe", tubePieces([[-190, 70, 2], [180, 70, 2]], 4, P, { maxLength: 80, spacing: 1 }).map((piece) => tubeSvg(piece, { tone: "mid" })).join(""), [0, 0, -60]);
  f.part("bsEx", balanceShaft(f, 70, 22, 0), [330, 0, 0]);
  f.part("bsIn", balanceShaft(f, -70, -22, 180), [330, 0, 0]);
  // The block's front face as a phantom, so the shafts inside it still show.
  f.part("block", `<g class="rb-ghost">${S(k.slabOf(plate, -70, 140, P, 4, 0.8), { tone: "lo" })}</g>` + L(k.sideRing([188, 70, 22], 16, "x", P, 32) + k.sideRing([188, -70, -22], 16, "x", P, 32), { tone: "lo" }));
  // Mounting pin and the intermediate sprocket with its paint mark, between the intake shaft and the crank.
  const PIN = X([188, -30, -60]);
  f.part("pin", S(G.disc(0, 26, 8, PIN, P), { tone: "hi" }) + S(G.ringBand(2, 5, 8, 10.5, PIN, P), { tone: "mid" }), [120, 0, 0]);
  const GEAR = X([206, -30, -60]);
  f.part("idler", S(G.disc(0, 10, 28, GEAR, P), { tone: "hi" }) + D(G.dotsOf(GEAR, 10, 26, 30, P, { all: true }), { size: 0.45, tone: "lo" }) + S(G.disc(10, 13, 10, GEAR, P), { tone: "mid" }) + L(G.spokesOf(GEAR, 10, 18, 27, 1, P, { phase: 0.4 }), { tone: "hi", className: "rb-mark" }), [180, 0, 0]);
  const SEAL = X([188, -70, -22]);
  f.part("seal", S(G.ringBand(0, 7, 16, 24, SEAL, P), { tone: "mid" }) + L(G.circleOf(SEAL, 7, 20, P, 40), { tone: "lo" }), [100, 0, 0]);
  return f.svg();
}

// ---------------------------------------------------------------- rear sealing flange
function flange() {
  const f = figure({ id: "flange", label: "The transmission-side sealing flange drawn off the back of the block: its seal ring round the crank journal and the sealant bead on its face.", fit: [[-260, -110, -120], [-90, -110, -120], [-260, 110, 120], [-90, 110, 120]], azimuth: 140, elevation: 24, width: 560, height: 420 });
  const { P, V } = f;
  // Seen from the transmission end: the block's rear face, the crank flange, then the sealing flange.
  f.part("block", S(k.slabOf({ x: -215, y: -100, w: 10, d: 200, r: 6 }, -95, 190, P, 4, 1), { tone: "lo" }) + S(G.disc(-8, 0, 42, X([-205, 0, 0]), P), { tone: "mid" }));
  const outline = [];
  for (let t = 0; t < 360; t += 10) {
    const r = 74 + 10 * Math.max(0, Math.cos((t - 270) * DEG));
    outline.push([r * Math.cos(t * DEG), r * Math.sin(t * DEG)]);
  }
  const plate = hull(outline);
  const F = X([-205, 0, 0]);
  const bead = plate.map(([y, z]) => [-205.5, y * 0.88, z * 0.88]);
  const beadSvg = tubePieces(bead, 1.3, P, { closed: true, maxLength: 300, spacing: 0.6 }).map((piece) => tubeSvg(piece, { tone: "lo" })).join("");
  const seal = S(G.ringBand(-14, -6, 44, 52, F, P), { tone: "hi" }) + L(G.circleOf(F, -14, 48, P, 48), { tone: "lo" });
  const plateSvg = S(G.prismOf(F, plate, -12, -6, P), { tone: "mid" });
  const holes = plate.filter((_, i) => i % 6 === 0).map(([y, z]) => k.iso([-218, y * 0.9, z * 0.9], P));
  f.part("flange", plateSvg + seal + beadSvg + D(holes, { size: 0.8, tone: "mid" }), [-80, 0, 0]);
  void V;
  return f.svg();
}

// ---------------------------------------------------------------- bore gauge
function bore() {
  const f = figure({ id: "bore", label: "A cylinder bore seen from above with its three measuring planes, top, middle and bottom, and the two directions A across the engine and B along it.", fit: [[-70, -70, -150], [70, 70, -150], [-70, -70, 0], [70, 70, 0]], azimuth: 34, elevation: 36 });
  const { P } = f;
  const plan = { x: -66, y: -66, w: 132, d: 132, r: 6 };
  const deck = k.slabOf(plan, -140, 140, P, 4, 1.2);
  const hole = k.ring(0, 0, BORE, 0, P, 72);
  deck.fill += hole;
  deck.top += hole;
  const F = Z([0, 0, 0]);
  const gauge = [-12, -60, -110].flatMap((s) => [0, 90].map((a) => k.iso(G.pointOf(F, s, BORE, (a + 180) * DEG), P)));
  // Inside the bore you only see what the opening shows: clip the planes to it.
  // The block drawn as a phantom so all three measuring planes inside the bore show.
  const full = [-12, -60, -110].map((s) => G.circleOf(F, s, BORE, P, 72)).join("");
  f.part("block", `<g class="rb-holed rb-ghost">${S(deck, { tone: "mid" })}</g>` + L(full, { tone: "hi", className: "rb-plane" }) + L([0, 90].map((a) => k.pathOf([-12, -110].map((s) => k.iso(G.pointOf(F, s, BORE, (a + 180) * DEG), P)))).join(""), { tone: "lo" }) + D(gauge, { size: 1.2, tone: "hi" }) + L(hole, { tone: "lo" }));
  return f.svg();
}

// ---------------------------------------------------------------- plastigauge
function plastigauge() {
  const f = figure({ id: "plastigauge", label: "A crankshaft journal with a strip of Plastigauge squashed across its full width, and the Plastigauge scale card to read the width against.", fit: [[-60, -60, -40], [60, 120, -40], [-60, -60, 60], [60, 120, 60]], azimuth: 30, elevation: 32 });
  const { P } = f;
  const F = X([0, 0, 0]);
  f.part("crank", S(G.disc(-24, 24, 29, F, P, { steps: 64 }), { tone: "hi" }));
  // The strip: a thin flat band over the top of the journal.
  const strip = arcSolids(F, 29, 29.6, 80, 100, -22, 22, P, f.V, { tone: "lo" });
  f.part("strip", strip[0][1]);
  // Scale card with bars of rising width.
  const card = { x: -40, y: 50, w: 80, d: 50, r: 2 };
  const bars = [2, 3, 4, 5, 6].map((w, i) => S(k.slabOf({ x: -30 + i * 14, y: 58, w: w * 1.4, d: 30, r: 0.2 }, 1.2, 0.2, P, 2), { tone: "lo", crease: "none" })).join("");
  f.part("card", S(k.slabOf(card, 0, 1.2, P, 3), { tone: "hi", crease: "none" }) + bars);
  return f.svg();
}

// ---------------------------------------------------------------- end play dial gauge
function endplay() {
  const f = figure({ id: "endplay", label: "A dial gauge on a holder bolted to the block, its plunger against a crank counterweight, to read the crankshaft's end play.", fit: [[-80, -80, -80], [140, 80, -80], [-80, -80, 120], [140, 80, 120]], azimuth: 36, elevation: 26 });
  const { P, V } = f;
  const F = X([0, 0, 0]);
  const crank = [[G.dot3([0, 0, 0], V), S(G.disc(-40, 0, 29, F, P), { tone: "hi" })], [G.dot3([8, 0, 0], V), S(G.prismOf(F, web(), 0, 16, P), { tone: "mid" })]];
  f.part("crank", f.sorted(crank));
  // Holder on the block, a post, an arm, and the dial; its plunger runs along x to the web face.
  const base = { x: 70, y: -45, w: 40, d: 30, r: 3 };
  const AX = X([16, 0, -20]);
  const DIAL = Y([70, -6, -20]);
  const arm = { x: 66, y: -34, w: 34, d: 8, r: 2 };
  const gauge =
    S(k.slabOf(base, -70, 18, P, 3, 0.6), { tone: "mid" }) +
    S(k.cylinder(90, -30, 4, -52, 40, P, 16), { tone: "mid", crease: "none" }) +
    S(k.slabOf(arm, -16, 6, P, 2, 0.3), { tone: "mid" }) +
    S(G.disc(0, 32, 2, AX, P), { tone: "hi" }) +
    S(G.disc(32, 40, 5, AX, P), { tone: "mid" }) +
    S(G.disc(-5, 12, 22, DIAL, P, { steps: 48 }), { tone: "hi" }) +
    L(G.circleOf(DIAL, 12, 19, P, 48) + G.circleOf(DIAL, 12, 2, P, 12), { tone: "lo" }) +
    L(k.pathOf([k.iso([70, 6, -20], P), k.iso([70 + 10, 6, -20 + 12], P)]), { tone: "hi", free: "end" });
  f.part("gauge", gauge, [40, 0, 0]);
  return f.svg();
}

// ---------------------------------------------------------------- oil spray jet
function jets() {
  const f = figure({ id: "block", label: "A piston oil spray jet and its pressure relief valve on the main bearing web, aimed up at the underside of the piston.", fit: [[-70, -70, -60], [70, 70, -60], [-70, -70, 140], [70, 70, 140]], azimuth: 34, elevation: 24 });
  const { P } = f;
  f.part("block", S(k.slabOf({ x: -60, y: -60, w: 120, d: 50, r: 4 }, -50, 30, P, 4, 1), { tone: "lo" }));
  f.part("piston", pistonBody(f, 70));
  const route = G.fillet([[10, -20, -20], [10, 0, -20], [10, 8, 10], [10, 8, 60]], 10);
  const jet = tubePieces(route, 3, P, { maxLength: 60, spacing: 1 }).map((piece) => tubeSvg(piece, { tone: "hi" })).join("");
  f.part("jets", S(k.slabOf({ x: 0, y: -34, w: 20, d: 20, r: 2 }, -20, 12, P, 3, 0.5), { tone: "mid" }) + jet + S(k.cylinder(10, -24, 6, -8, 6, P, 6, 0.4), { tone: "hi" }), [0, 30, 0]);
  return f.svg();
}

// ---------------------------------------------------------------- overview: the bare block
function overview() {
  const f = figure({ id: "overview", label: "The bare cylinder block: four 82.5 mm bores at 88 mm centres, the crankcase below.", fit: [[-210, -100, -60], [195, -100, -60], [-210, 100, 225], [195, 100, 225]], azimuth: 32, elevation: 28 });
  const { P } = f;
  const upper = { x: -205, y: -70, w: 393, d: 140, r: 6 };
  const lower = { x: -205, y: -98, w: 393, d: 196, r: 6 };
  const holes = [132, 44, -44, -132].map((x) => k.ring(x, 0, BORE, 221, P, 72)).join("");
  const deck = k.slabOf(upper, 120, 101, P, 4, 1.2);
  deck.fill += holes;
  deck.top += holes;
  const flare = G.prismOf(X([0, 0, 0]), [[-98, 40], [98, 40], [70, 120], [-70, 120]], -205, 188, P);
  f.part("block", S(k.slabOf(lower, -58, 98, P, 4, 1), { tone: "mid", inner: L(k.sideSeam(lower, -49, P, 4), { tone: "faint" }) }) + S(flare, { tone: "mid" }) + `<path class="rb-bore" d="${holes}"/>` + `<g class="rb-holed">${S(deck, { tone: "mid" })}</g>` + L(holes, { tone: "lo" }) + L([-176, -88, 0, 88].map((x) => k.segment([x, 70, 120], [x, 70, 221], P)).join(""), { tone: "faint" }));
  return f.svg();
}

export function buildFigures() {
  return { overview: overview(), bore: bore(), piston: piston(), rings: rings(), block: jets(), balance: balance(), mains: mains(), plastigauge: plastigauge(), endplay: endplay(), rod: rod(), flange: flange() };
}
