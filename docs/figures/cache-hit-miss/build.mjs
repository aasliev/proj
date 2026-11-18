// Fig: cache hit vs. miss, drawn as a tool crib with an overhead shuttle.
// Build: node docs/figures/cache-hit-miss/build.mjs [--light]
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../../.claude/skills/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../../../.claude/skills/anatomy/scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const OUT = theme === "light" ? "cache-hit-miss-light.html" : "cache-hit-miss.html";

const WIDTH = 640;
const HEIGHT = 400;

// ---- World layout (units ≈ px; x along the beam, y toward the front, z up) ----
const PLATE = { x: -182, y: -40, w: 364, d: 104, r: 8 };
const PLATE_H = 12;
const FOOT_H = 5;

const POSTS = [-170, 170];
const POST_R = 4;
const BEAM = { from: -178, to: 178 };
const FLANGE_LO = 80; // underside of the beam's bottom flange
const BEAM_TOP = 90;
const TROLLEY = { z: 64, h: 12, w: 22, d: 18 };

const COUNTER = { x: -158, y: -20, w: 46, d: 40, r: 3 };
const COUNTER_H = 14;
const HOME = -135; // shuttle parks over the counter
const WATCH = [HOME, 2];
const WATCH_R = 13;
const WATCH_TOP = COUNTER_H + 6;

const TRAY = { x: -110, y: -18, w: 80, d: 36, r: 3 };
const TRAY_H = 9;
const SLOTS = [-95, -70, -45];

const BASE = { x: -15, y: -18, w: 150, d: 36, r: 3 };
const BASE_H = 4;
const MAGS = [0, 24, 48, 72, 96, 120];
const MAG_OUT = 11; // half outer size
const MAG_WALL = 2;
const MAG_RIM = BASE_H + 22;
const TILE = 16;
const TILE_H = 5;
const REST_TOP = MAG_RIM - 2; // a waiting tile sits 2 below the rim
const LIFT = MAG_RIM + TILE_H - REST_TOP; // the feed lifts it clear of the rim

const TRAVEL_Z = 44; // shuttle head height while it travels
const HEAD_TOP = 12;

const P = k.fitProjection(
  [
    ...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0),
    [BEAM.from, -8, BEAM_TOP],
    [BEAM.to, 8, BEAM_TOP],
  ],
  WIDTH,
  HEIGHT,
  { pad: 22, azimuth: 64 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// ---- Parts ----

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));

  // Ruler along the front edge: the shuttle's travel, one tick per 5 units.
  const strip = { x: -104, y: PLATE.y + PLATE.d - 15, w: 266, d: 7, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 5, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));

  // Label plate in front of the counter: three rule lines of printed text.
  const label = { x: -158, y: 30, w: 34, d: 16, r: 1.2 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P, 4), { tone: "faint" }));
  [[22, 0], [16, 4], [19, 8]].forEach(([len, dy]) => {
    out.push(L(k.lineOnTop([label.x + 5, label.y + 4 + dy], [label.x + 5 + len, label.y + 4 + dy], 0.8, P), { tone: "lo", free: true }));
  });
  out.push(D([at([label.x + 2.6, label.y + 2.6, 0.8]), at([label.x + label.w - 2.6, label.y + label.d - 2.6, 0.8])], { size: 0.4 }));
  return out.join("");
}

function post(x) {
  const out = [];
  out.push(S(k.cylinder(x, 0, 7, 0, 2, P, 28), { tone: "lo" }));
  out.push(D(k.circleRing(x, 0, 5.4, 4).map(([px, py]) => at([px, py, 2])), { size: 0.4 }));
  out.push(S(k.cylinder(x, 0, 5.2, 2, 3, P, 28), { tone: "mid" }));
  out.push(S(k.cylinder(x, 0, POST_R, 5, FLANGE_LO - 7, P, 24), { tone: "mid", crease: "none" }));
  out.push(L([28, 52].map((z) => k.sideArc(x, 0, POST_R, z, P, 16)).join(""), { tone: "faint" }));
  out.push(S(k.slabOf({ x: x - 7, y: -7, w: 14, d: 14, r: 1.5 }, FLANGE_LO - 2, 2, P, 3, 0.4), { tone: "mid" }));
  return out.join("");
}

function counter() {
  const out = [`<path class="iso-halo" d="${k.haloOf(COUNTER, 0, 1, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(COUNTER, 0, 3, P, 5, 0.4), { tone: "lo" }));
  out.push(S(k.slabOf(k.insetPlan(COUNTER, 1.5), 3, COUNTER_H - 3, P, 5, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(COUNTER, 4), COUNTER_H, P, 5), { tone: "faint" }));
  out.push(D(k.corners(COUNTER, 5).map(([x, y]) => at([x, y, COUNTER_H])), { size: 0.45 }));
  // Stopwatch, face up: one turn is 100 ms of the model.
  const [x, y] = WATCH;
  out.push(S(k.cylinder(x, y, WATCH_R + 1.5, COUNTER_H, 1.5, P, 48), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, WATCH_R, COUNTER_H + 1.5, WATCH_TOP - COUNTER_H - 1.5, P, 64, 0.6), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, WATCH_R, WATCH_TOP - 2.6, WATCH_TOP, 72, P, 0.3), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, WATCH_R, WATCH_TOP - 2.6, P, 40), { tone: "faint" }));
  out.push(L(k.ring(x, y, WATCH_R - 1.8, WATCH_TOP, P, 64), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, WATCH_R - 2.4, 50, 5, WATCH_TOP, [1, 2.2], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.ring(x, y, 1.4, WATCH_TOP, P, 16), { tone: "mid" }));
  // Crown and reset pusher on the far side.
  out.push(S(k.slabOf({ x: x - 2, y: y - WATCH_R - 3.4, w: 4, d: 4, r: 1 }, COUNTER_H + 1.6, 3, P, 3), { tone: "mid" }));
  out.push(S(k.cylinder(x, y - WATCH_R - 4.4, 2.4, COUNTER_H + 1, 4.2, P, 20), { tone: "mid" }));
  out.push(L(k.knurl(x, y - WATCH_R - 4.4, 2.4, COUNTER_H + 1, COUNTER_H + 5.2, 16, P), { tone: "lo" }));
  out.push(k.faceTextSvg(k.topMatrix([x, y + 6.5], WATCH_TOP, P, "x"), "ms", { size: 2.2, tone: "lo", anchor: "middle" }));
  return out.join("");
}

function tray() {
  const out = [`<path class="iso-halo" d="${k.haloOf(TRAY, 0, 1, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(TRAY, 0, 3, P, 5, 0.4), { tone: "lo" }));
  out.push(S(k.slabOf(k.insetPlan(TRAY, 1.5), 3, TRAY_H - 3, P, 5, 0.8), { tone: "mid" }));
  out.push(D(k.corners(TRAY, 4).map(([x, y]) => at([x, y, TRAY_H])), { size: 0.45 }));
  for (const sx of SLOTS) {
    // Trapdoor that drops an evicted tile, and four locating pins.
    out.push(L(k.planOutline({ x: sx - 9, y: -9, w: 18, d: 18, r: 1.5 }, TRAY_H, P, 3), { tone: "lo" }));
    out.push(L(k.lineOnTop([sx, -9], [sx, 9], TRAY_H, P), { tone: "faint" }));
    for (const [px, py] of byDepth(k.corners({ x: sx - 12, y: -12, w: 24, d: 24 }, 0))) {
      out.push(S(k.cylinder(px, py, 1.1, TRAY_H, 1.6, P, 12), { tone: "mid", crease: "none" }));
    }
  }
  // Index marks on the front face, one under each slot.
  const front = TRAY.y + TRAY.d - 1.5;
  out.push(L(SLOTS.map((sx) => k.segment([sx, front, TRAY_H - 2.5], [sx, front, TRAY_H], P)).join(""), { tone: "lo" }));
  return out.join("");
}

// A tile is a part: its pips (1–6) are its key. Built at the origin, top at z = 0.
const PIPS = {
  1: [[0, 0]],
  2: [[-4, -4], [4, 4]],
  3: [[-4, -4], [0, 0], [4, 4]],
  4: [[-4, -4], [4, -4], [-4, 4], [4, 4]],
  5: [[-4, -4], [4, -4], [0, 0], [-4, 4], [4, 4]],
  6: [[-4, -4], [4, -4], [-4, 0], [4, 0], [-4, 4], [4, 4]],
};
function tile(key) {
  const plan = { x: -TILE / 2, y: -TILE / 2, w: TILE, d: TILE, r: 1.4 };
  return (
    S(k.slabOf(plan, -TILE_H, TILE_H, P, 3, 0.5), { tone: "mid" }) +
    L(k.planOutline(k.insetPlan(plan, 1.6), 0, P, 3), { tone: "faint" }) +
    D(PIPS[key].map(([x, y]) => at([x, y, 0])), { size: 0.85, tone: "hi" })
  );
}
const tileAt = (key, [x, y, top], extra = "") => `<g data-tile="${key}"${extra} transform="${k.translateAlong(P, [x, y, top])}">${tile(key)}</g>`;

function slotGroups() {
  return SLOTS.map((sx, s) => {
    const prism = [
      at([sx - 9, -9, TRAY_H]), at([sx + 9, -9, TRAY_H]), at([sx + 9, 9, TRAY_H]), at([sx - 9, 9, TRAY_H]),
      at([sx - 9, -9, 400]), at([sx + 9, -9, 400]), at([sx + 9, 9, 400]), at([sx - 9, 9, 400]),
    ];
    const hull = convexHull(prism);
    const tiles = [1, 2, 3, 4, 5, 6].map((key) => tileAt(key, [sx, 0, TRAY_H + TILE_H], ' style="display:none"')).join("");
    return `<clipPath id="slot-${s}"><path d="${k.pathOf(hull, true)}"/></clipPath><g data-slot="${s}" clip-path="url(#slot-${s})"><g data-sink>${tiles}</g></g>`;
  }).join("");
}

function convexHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  for (const p of pts.reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function originBase() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, 0, 1, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(BASE, 0, BASE_H, P, 5, 0.6), { tone: "lo" }));
  out.push(D(k.corners(BASE, 3.5).map(([x, y]) => at([x, y, BASE_H])), { size: 0.45 }));
  return out.join("");
}

// A magazine is a square tube with a spring feed: a tile always waits 2 below the rim.
function magazine(cx, i) {
  const o = MAG_OUT, w = MAG_WALL, z = BASE_H, h = MAG_RIM - BASE_H;
  const wall = (plan) => S(k.slabOf(plan, z, h, P, 2, 0.3), { tone: "mid" });
  const back = wall({ x: cx - o, y: -o, w: 2 * o, d: w, r: 0.6 }) + wall({ x: cx - o, y: -o + w, w, d: 2 * o - 2 * w, r: 0.3 });
  const front = wall({ x: cx + o - w, y: -o + w, w, d: 2 * o - 2 * w, r: 0.3 }) + wall({ x: cx - o, y: o - w, w: 2 * o, d: w, r: 0.6 });
  // Sight slot on the front face with the stack of waiting tiles behind it.
  const fy = o;
  const slot = [[cx - 2, fy, z + 3], [cx + 2, fy, z + 3], [cx + 2, fy, MAG_RIM - 3], [cx - 2, fy, MAG_RIM - 3]];
  const window = L(k.pathOf(slot.map(at), true), { tone: "lo" });
  const seams = [];
  for (let zz = REST_TOP - TILE_H; zz > z + 3; zz -= TILE_H) seams.push(k.segment([cx - 2, fy, zz], [cx + 2, fy, zz], P));
  const stack = L(seams.join(""), { tone: "faint" });
  const screws = D([at([cx - o + 3, fy, z + 2.5]), at([cx + o - 3, fy, z + 2.5])], { size: 0.4 });
  const lift = `<g data-lift="${i}">${tileAt(i + 1, [cx, 0, REST_TOP])}</g>`;
  return `<g data-key="${i + 1}" class="cc-pick">${back}${lift}${front}${window}${stack}${screws}</g>`;
}

function head() {
  // Built with its suction cup's face at z = 0, centred on x = 0.
  const out = [];
  out.push(S(k.cylinder(0, 0, 5, 0, 1.6, P, 28), { tone: "lo" }));
  out.push(S(k.cylinder(0, 0, 6.2, 1.6, 4, P, 32, 0.4), { tone: "mid" }));
  out.push(L(k.sideArc(0, 0, 6.2, 3.6, P, 20), { tone: "faint" }));
  out.push(S(k.cylinder(0, 0, 3.6, 5.6, 2.4, P, 24), { tone: "mid" }));
  out.push(S(k.slabOf({ x: -6, y: -3.5, w: 12, d: 7, r: 1 }, 8, HEAD_TOP - 8, P, 3, 0.4), { tone: "mid" }));
  out.push(D([at([-4, 0, HEAD_TOP]), at([4, 0, HEAD_TOP])], { size: 0.5 }));
  // Vacuum hose spigot on the near side.
  out.push(S(k.cylinder(4.6, 4.2, 1.2, 8.6, 2.6, P, 12), { tone: "mid", crease: "none" }));
  return out.join("");
}

function trolleyBack() {
  return S(k.slabOf({ x: -9, y: -TROLLEY.d / 2 - 1.6, w: 18, d: 1.6, r: 0.4 }, TROLLEY.z + 4, BEAM_TOP - 4 - TROLLEY.z - 4, P, 2, 0.3), { tone: "mid" });
}
function trolleyBody() {
  const out = [];
  const body = { x: -TROLLEY.w / 2, y: -TROLLEY.d / 2, w: TROLLEY.w, d: TROLLEY.d, r: 2 };
  out.push(S(k.slabOf(body, TROLLEY.z, TROLLEY.h, P, 4, 0.6), { tone: "mid" }));
  out.push(L(k.sideSeam(body, TROLLEY.z + 4, P, 4), { tone: "faint" }));
  // Two cable drums show as round covers on the front face.
  for (const x of [-5, 5]) out.push(L(k.sideRing([x, TROLLEY.d / 2, TROLLEY.z + 8.5], 2.4, "y", P, 20), { tone: "lo" }));
  out.push(D([-5, 5].map((x) => at([x, TROLLEY.d / 2, TROLLEY.z + 8.5])), { size: 0.45 }));
  return out.join("");
}
function trolleyFront() {
  const out = [];
  out.push(S(k.slabOf({ x: -9, y: TROLLEY.d / 2, w: 18, d: 1.6, r: 0.4 }, TROLLEY.z + 4, BEAM_TOP - 4 - TROLLEY.z - 4, P, 2, 0.3), { tone: "mid" }));
  const face = TROLLEY.d / 2 + 1.6;
  for (const x of [-5, 5]) {
    out.push(L(k.sideRing([x, face, FLANGE_LO + 4.2], 2.2, "y", P, 18), { tone: "lo" }));
  }
  out.push(D([-5, 5].map((x) => at([x, face, FLANGE_LO + 4.2])), { size: 0.5, tone: "hi" }));
  return out.join("");
}

function beam() {
  const out = [];
  const len = BEAM.to - BEAM.from;
  out.push(S(k.slabOf({ x: BEAM.from, y: -7, w: len, d: 14, r: 1 }, FLANGE_LO, 2, P, 2, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: BEAM.from, y: -1.5, w: len, d: 3, r: 0.4 }, FLANGE_LO + 2, 6, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.slabOf({ x: BEAM.from, y: -7, w: len, d: 14, r: 1 }, FLANGE_LO + 8, 2, P, 2, 0.4), { tone: "mid" }));
  out.push(D([BEAM.from + 4, BEAM.to - 4].flatMap((x) => [at([x, -4, BEAM_TOP]), at([x, 4, BEAM_TOP])]), { size: 0.45 }));
  // End stops: rubber buffers on the web at both ends.
  for (const x of [BEAM.from + 4, BEAM.to - 10]) {
    out.push(S(k.slabOf({ x, y: 1.5, w: 6, d: 3.5, r: 1 }, FLANGE_LO + 2.5, 5, P, 3), { tone: "lo" }));
  }
  return out.join("");
}

// ---- Assemble in painter's order ----
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "A tool crib that explains a cache. An overhead shuttle runs on a beam between a counter with a stopwatch, a three-slot cache tray next to it, and a far rack of six magazines, the origin. On a hit the shuttle touches the part already in the cache tray and returns. On a miss it travels to the far magazine, waits while the part is fed up, carries it back, drops the least recently used part through a trapdoor and seats the new one.",
  body: [
    plate(),
    post(POSTS[0]),
    counter(),
    `<path data-part="needle" class="iso-line" data-tone="hi" data-free="start" d="" style="stroke-width:1"/>`,
    tray(),
    slotGroups(),
    originBase(),
    MAGS.map((cx, i) => magazine(cx, i)).join(""),
    post(POSTS[1]),
    `<path data-part="cables" class="iso-line" data-tone="lo" d=""/>`,
    `<g data-part="head">${head()}<g data-carry>${[1, 2, 3, 4, 5, 6].map((key) => tileAt(key, [0, 0, 0], ' style="display:none"')).join("")}</g></g>`,
    `<g data-part="trolley-back">${trolleyBack()}</g>`,
    `<g data-part="trolley-body">${trolleyBody()}</g>`,
    beam(),
    `<g data-part="trolley-front">${trolleyFront()}</g>`,
  ],
});

// ---- Live model ----
const live = `
const P = ${JSON.stringify(P)};
const SLOTS = ${JSON.stringify(SLOTS)}, MAGS = ${JSON.stringify(MAGS)};
const HOME = ${HOME}, TRAVEL_Z = ${TRAVEL_Z}, HEAD_TOP = ${HEAD_TOP}, TROLLEY_Z = ${TROLLEY.z};
const SLOT_TOP = ${TRAY_H + TILE_H}, MAG_PICK = ${REST_TOP + LIFT}, LIFT = ${LIFT};
const WATCH = ${JSON.stringify(WATCH)}, WATCH_R = ${WATCH_R}, WATCH_TOP = ${WATCH_TOP};
// The model, in milliseconds: the shuttle covers 1 unit in 0.08 ms; the origin takes 40 ms to produce a part.
const MS_PER_UNIT = 0.08, ORIGIN_MS = 40, GRIP_MS = 1, EVICT_MS = 5, REAL_PER_MS = 40, PER_TURN = 100;

const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const $ = (sel) => svg.querySelector(sel);
const headEl = $('[data-part="head"]'), cables = $('[data-part="cables"]'), needle = $('[data-part="needle"]');
const trolley = ["trolley-back", "trolley-body", "trolley-front"].map((n) => $('[data-part="' + n + '"]'));
const carryTiles = [...$('[data-carry]').children];
const slotEls = SLOTS.map((_, s) => $('[data-slot="' + s + '"]'));
const liftEls = MAGS.map((_, i) => $('[data-lift="' + i + '"]'));
const readout = document.querySelector("[data-readout]");
const buttons = [...document.querySelectorAll("[data-request]")];
const still = matchMedia("(prefers-reduced-motion: reduce)");

const state = { gx: HOME, gz: TRAVEL_Z, carry: 0, lift: [0, 0, 0, 0, 0, 0], sink: [0, 0, 0], slots: [1, 2, 0], used: [2, 1, 0], lit: 0, litSlot: -1, clock: 0 };
let stamp = 3, hits = 0, total = 0;
let plan = null, step = 0, stepT = 0, from = null, startClock = 0, said = "ready · cache holds parts 1 and 2";
const queue = [];

const ease = (u) => u * u * (3 - 2 * u);
const travel = (dx) => Math.abs(dx) * MS_PER_UNIT;

function planFor(key) {
  const s0 = state.slots.indexOf(key);
  const steps = [];
  const go = (gx, say) => steps.push({ ms: () => travel(gx - state.gx), to: { gx }, say });
  const dip = (gz, say) => steps.push({ ms: () => travel(gz - state.gz), to: { gz }, say });
  if (s0 >= 0) {
    const x = SLOTS[s0];
    go(x, "part " + key + " · in the cache");
    dip(SLOT_TOP, "part " + key + " · hit · reading");
    steps.push({ ms: () => GRIP_MS, at: () => { state.litSlot = s0; state.lit = key; }, say: "part " + key + " · hit · reading" });
    dip(TRAVEL_Z, "part " + key + " · hit · returning");
    go(HOME, "part " + key + " · hit · returning");
    steps.push({ ms: () => 0, at: () => { state.used[s0] = stamp++; hits++; total++; }, done: "hit" });
    return steps;
  }
  const m = key - 1;
  // Fill an empty slot, else evict the least recently used one.
  let s = state.slots.indexOf(0);
  if (s < 0) s = state.used.indexOf(Math.min(...state.used));
  go(MAGS[m], "part " + key + " · miss · to origin");
  steps.push({ ms: () => ORIGIN_MS, to: { lift: [m, 1] }, say: "part " + key + " · miss · origin producing" });
  dip(MAG_PICK, "part " + key + " · miss · picking");
  steps.push({ ms: () => GRIP_MS, at: () => { state.carry = key; state.lit = key; state.litSlot = -1; state.lift[m] = 0; }, say: "part " + key + " · miss · picking" });
  dip(TRAVEL_Z, "part " + key + " · miss · carrying back");
  go(SLOTS[s], "part " + key + " · miss · carrying back");
  if (state.slots[s]) {
    const old = state.slots[s];
    steps.push({ ms: () => EVICT_MS, to: { sink: [s, 1] }, say: "evicting part " + old + " (least recently used)" });
    steps.push({ ms: () => 0, at: () => { state.slots[s] = 0; state.sink[s] = 0; } });
  }
  dip(SLOT_TOP, "part " + key + " · miss · filling cache");
  steps.push({ ms: () => GRIP_MS, at: () => { state.slots[s] = key; state.used[s] = stamp++; state.carry = 0; state.litSlot = s; }, say: "part " + key + " · miss · filling cache" });
  dip(TRAVEL_Z, "part " + key + " · miss · returning");
  go(HOME, "part " + key + " · miss · returning");
  steps.push({ ms: () => 0, at: () => { total++; }, done: "miss" });
  return steps;
}

function begin(key) {
  plan = planFor(key); step = 0; stepT = 0; from = null;
  state.clock = 0; state.key = key;
}

function advance(ms) {
  while (plan && ms >= 0) {
    const st = plan[step];
    if (!from) from = { gx: state.gx, gz: state.gz, lift: st.to && st.to.lift ? state.lift[st.to.lift[0]] : 0, sink: st.to && st.to.sink ? state.sink[st.to.sink[0]] : 0, dur: st.ms() };
    if (st.say) said = st.say;
    const left = from.dur - stepT;
    const use = Math.min(ms, left);
    stepT += use; state.clock += use; ms -= use;
    const u = from.dur > 0 ? ease(Math.min(1, stepT / from.dur)) : 1;
    if (st.to) {
      if ("gx" in st.to) state.gx = from.gx + (st.to.gx - from.gx) * u;
      if ("gz" in st.to) state.gz = from.gz + (st.to.gz - from.gz) * u;
      if (st.to.lift) state.lift[st.to.lift[0]] = from.lift + (st.to.lift[1] - from.lift) * u;
      if (st.to.sink) state.sink[st.to.sink[0]] = from.sink + (st.to.sink[1] - from.sink) * u;
    }
    if (stepT >= from.dur - 1e-9) {
      if (st.at) st.at();
      if (st.done) {
        said = "part " + state.key + " · " + st.done + " · " + Math.round(state.clock) + " ms · hit rate " + hits + "/" + total;
        stage.setAttribute("aria-valuetext", said);
        plan = null;
        break;
      }
      step++; stepT = 0; from = null;
      if (ms <= 0) break;
    } else break;
  }
}

const needleAt = (ms) => {
  const a = (ms / PER_TURN) * Math.PI * 2;
  const [x, y] = WATCH;
  return lineOnTop([x - Math.sin(a) * 2.6, y + Math.cos(a) * 2.6], [x + Math.sin(a) * (WATCH_R - 3.2), y - Math.cos(a) * (WATCH_R - 3.2)], WATCH_TOP + 0.2, P);
};
const show = (el, on) => { const v = on ? "" : "none"; if (el.style.display !== v) el.style.display = v; };
const setLit = (g, on) => { for (const s of g.querySelectorAll(".iso-solid")) { if (on) s.setAttribute("data-lit", ""); else s.removeAttribute("data-lit"); } };

function draw() {
  headEl.setAttribute("transform", translateAlong(P, [state.gx, 0, state.gz]));
  for (const t of trolley) t.setAttribute("transform", translateAlong(P, [state.gx, 0, 0]));
  cables.setAttribute("d", [-3, 3].map((dx) => segment([state.gx + dx, 0, state.gz + HEAD_TOP], [state.gx + dx, 0, TROLLEY_Z], P)).join(""));
  carryTiles.forEach((t, i) => { show(t, state.carry === i + 1); setLit(t, state.carry === i + 1); });
  slotEls.forEach((el, s) => {
    const sink = el.firstElementChild;
    sink.setAttribute("transform", translateAlong(P, [0, 0, -state.sink[s] * 7]));
    sink.style.opacity = String(1 - state.sink[s] * 0.6);
    [...sink.children].forEach((t, i) => { show(t, state.slots[s] === i + 1); setLit(t, state.litSlot === s && state.slots[s] === i + 1); });
  });
  liftEls.forEach((el, i) => el.setAttribute("transform", translateAlong(P, [0, 0, state.lift[i] * LIFT])));
  needle.setAttribute("d", needleAt(state.clock));
  if (readout.textContent !== said) readout.textContent = said;
  const held = new Set(state.slots);
  buttons.forEach((b) => b.toggleAttribute("data-cached", held.has(Number(b.dataset.request))));
}

// Idle tour: a mix of repeats (hits) and new parts (misses).
const TOUR = [1, 4, 2, 4, 5, 1, 4, 6, 2, 6, 3, 6];
let tourAt = 0, touring = !still.matches, idleTimer = 0, gap = 0.9, frame = 0, last = 0, visible = false;

function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (!plan) {
    if (queue.length) begin(queue.shift());
    else if (touring) { gap -= dt; if (gap <= 0) { begin(TOUR[tourAt++ % TOUR.length]); gap = 1.4; } }
  }
  if (plan) advance(still.matches ? 1e9 : (dt * 1000) / REAL_PER_MS);
  draw();
  const busy = plan || queue.length || touring;
  frame = visible && busy ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
function request(key) {
  touring = false;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => { touring = !still.matches; gap = 0.4; run(); }, 6000);
  if (queue.length < 3) queue.push(key);
  run();
}
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
svg.addEventListener("click", (event) => {
  const hit = event.target.closest("[data-key]");
  if (hit) request(Number(hit.dataset.key));
});
buttons.forEach((b) => b.addEventListener("click", () => request(Number(b.dataset.request))));
stage.addEventListener("keydown", (event) => {
  const n = Number(event.key);
  if (n >= 1 && n <= 6) { event.preventDefault(); request(n); }
});
draw();
`;

const controls = `<div class="cc-controls" role="group" aria-label="Request a part">${[1, 2, 3, 4, 5, 6]
  .map((n) => `<button type="button" data-request="${n}" aria-label="Request part ${n}">${n}</button>`)
  .join("")}</div>`;

const css = `<style>
.cc-pick{cursor:pointer}
.cc-controls{display:flex;gap:6px;justify-content:center;margin:14px 0 2px}
.cc-controls button{font:12px/1 var(--anatomy-mono);color:var(--anatomy-text);background:transparent;border:1px solid var(--anatomy-lo);border-radius:6px;width:34px;height:28px;cursor:pointer;transition:border-color .2s var(--anatomy-ease),color .2s var(--anatomy-ease)}
.cc-controls button:hover{border-color:var(--anatomy-mid);color:var(--anatomy-strong)}
.cc-controls button[data-cached]{border-color:var(--anatomy-mid);color:var(--anatomy-strong)}
.cc-controls button:focus-visible,.iso-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
</style>`;

const body =
  css +
  k.plateHtml({
    fig: "Fig 1",
    title: "Cache hit vs. miss",
    hint: "Press 1–6 or click a magazine",
    readout: "ready · cache holds parts 1 and 2",
    keys: [
      { mark: "lit", label: "The requested part" },
      { mark: "raised", label: "Cache tray: three slots beside the counter" },
      { mark: "flat", label: "Origin: six magazines at the far end" },
    ],
    caption:
      "Every request starts at the counter. On a hit the shuttle only travels to the cache tray, about 12–20 ms. On a miss it travels to the origin, waits 40 ms while the part is fed up, carries it back into the cache (dropping the least recently used part when all three slots are full) and returns, about 70–95 ms. The stopwatch makes one turn per 100 ms. Shuttle travel is 0.08 ms per unit, and the figure runs 40 times slower than the model. Buttons outlined brighter are parts in the cache right now.",
    body: `<div class="iso-stage" tabindex="0" role="group" aria-label="Cache explainer: press 1 to 6 to request a part">${svg}</div>${controls}`,
  });

writeFileSync(join(HERE, OUT), k.pageHtml({ title: "Cache hit vs. miss", theme, body, script: kitScript() + live }));
console.log("wrote", OUT);
