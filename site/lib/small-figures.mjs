// Generic drawings for the small service jobs: a filter housing, a wiper, a battery and a
// headlamp. They show what the parts look like and how they come apart, not their exact shape
// or where they sit on a particular car.
import { figure, S, L, k, G, X, Z } from "./figure.mjs";
import { tubePieces, tubeSvg } from "../../.claude/skills/anatomy/kit/tube.mjs";

// A filter in its housing: box, pleated element, lid. `flat` makes it a thin cabin filter tray.
export function filterBox({ id, label, flat = false }) {
  const w = flat ? 240 : 260, d = flat ? 200 : 180, hBox = flat ? 14 : 70, hEl = flat ? 26 : 50;
  const f = figure({ id, label, fit: [[-150, -120, 0], [150, 120, 0], [-150, -120, 220], [150, 120, 220]], azimuth: 36, elevation: 34, width: 520, height: 400 });
  const { P } = f;
  const box = { x: -w / 2 - 12, y: -d / 2 - 12, w: w + 24, d: d + 24, r: 10 };
  f.part("housing", S(k.slabOf(box, 0, hBox, P, 4, 1), { tone: "lo" }) + (flat ? "" : S(k.cylinder(box.x + box.w + 20, 0, 34, 20, 40, P, 32, 0.4), { tone: "lo" })));
  const el = { x: -w / 2, y: -d / 2, w, d, r: 4 };
  const pleats = [];
  for (let i = 1; i < 18; i++) {
    const x = el.x + (w * i) / 18;
    pleats.push(k.pathOf([k.iso([x, el.y + 6, hBox + hEl], P), k.iso([x, el.y + d - 6, hBox + hEl], P)]));
  }
  f.part("element", S(k.slabOf(el, hBox, hEl, P, 4, 1), { tone: "hi" }) + L(pleats.join(""), { tone: "lo" }), [0, 0, 110]);
  const lidZ = hBox + hEl;
  f.part("lid", S(k.slabOf(box, lidZ, flat ? 6 : 40, P, 4, 1), { tone: "mid" }) + (flat ? S(k.slabOf({ x: box.x + box.w - 30, y: -20, w: 20, d: 40, r: 4 }, lidZ + 6, 4, P, 2, 0.5), { tone: "hi" }) : [-1, 1].map((s) => S(k.slabOf({ x: -20, y: s * (box.d / 2) - 6, w: 40, d: 12, r: 3 }, lidZ - 16, 30, P, 2, 0.5), { tone: "hi" })).join("")), [0, 0, 200]);
  return f.svg();
}

// A wiper arm and blade on the windscreen.
export function wiper() {
  const f = figure({ id: "wiper", label: "A wiper arm lying on the windscreen with its blade clipped to the end of the arm; the arm's spindle and nut cover at the bottom.", fit: [[-60, -260, 0], [60, 260, 0], [-60, -260, 120], [60, 260, 120]], azimuth: 62, elevation: 38, width: 560, height: 380 });
  const { P } = f;
  f.part("screen", `<g class="rb-ghost">${S(k.slabOf({ x: -80, y: -270, w: 120, d: 540, r: 6 }, 0, 4, P, 2, 0.5), { tone: "hi" })}</g>`);
  const F = G.frameOf([0, -240, 10], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  const arm = S(G.prismOf(F, [[-5, 0], [5, 0], [5, 8], [-5, 8]], 0, 420, P), { tone: "lo" }) + S(k.cylinder(0, -240, 18, 4, 18, P, 32, 0.4), { tone: "lo" });
  f.part("arm", arm, [0, 0, 70]);
  const Fb = G.frameOf([-22, -180, 4], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  f.part("blade", S(G.prismOf(Fb, [[-6, 0], [6, 0], [4, 10], [-4, 10]], 0, 420, P), { tone: "mid" }) + S(G.prismOf(Fb, [[-1.5, -4], [1.5, -4], [1.5, 0], [-1.5, 0]], 2, 418, P), { tone: "lo" }), [-90, 0, 40]);
  void X;
  return f.svg();
}

// A battery with its terminal clamps and hold-down.
export function battery() {
  const f = figure({ id: "battery", label: "A car battery: the case, the negative and positive terminal clamps on its posts, and the hold-down clamp at its base.", fit: [[-160, -100, 0], [160, 100, 0], [-160, -100, 260], [160, 100, 260]], azimuth: 34, elevation: 30, width: 520, height: 400 });
  const { P } = f;
  f.part("tray", S(k.slabOf({ x: -150, y: -95, w: 300, d: 190, r: 8 }, 0, 10, P, 4, 1), { tone: "lo" }));
  f.part("holdDown", S(k.slabOf({ x: -40, y: -112, w: 80, d: 24, r: 3 }, 10, 18, P, 2, 0.5), { tone: "lo" }), [0, -60, 0]);
  const body = S(k.slabOf({ x: -130, y: -85, w: 260, d: 170, r: 8 }, 10, 180, P, 4, 1), { tone: "mid" }) + S(k.slabOf({ x: -126, y: -81, w: 252, d: 162, r: 6 }, 190, 6, P, 4, 1), { tone: "hi" });
  const posts = [-90, 90].map((x) => S(k.cylinder(x, 40, 10, 196, 18, P, 24, 0.3), { tone: "hi" })).join("");
  f.part("battery", body + posts + L(k.pathOf([k.iso([82, 40, 230], P), k.iso([98, 40, 230], P)]) + k.pathOf([k.iso([90, 32, 230], P), k.iso([90, 48, 230], P)]) + k.pathOf([k.iso([-98, 40, 230], P), k.iso([-82, 40, 230], P)]), { tone: "lo" }), [0, 0, 0]);
  const clamp = (x) => S(k.slabOf({ x: x - 18, y: 22, w: 36, d: 50, r: 6 }, 204, 16, P, 2, 0.5), { tone: "lo" }) + S(k.cylinder(x, 70, 7, 210, 8, P, 16, 0.3), { tone: "lo" });
  f.part("negClamp", clamp(-90), [0, 0, 60]);
  f.part("posClamp", clamp(90) + S(k.slabOf({ x: 66, y: 16, w: 48, d: 62, r: 8 }, 220, 6, P, 2, 0.5), { tone: "hi" }), [0, 0, 60]);
  return f.svg();
}

// The back of a headlamp: housing, dust cover, bulb and its connector.
export function headlamp() {
  const f = figure({ id: "headlamp", label: "The back of a headlamp: the housing, the round dust cover, the bulb behind it and the wiring connector on the bulb.", fit: [[-40, -150, -80], [-40, 150, 120], [300, -150, -80], [300, 150, 120]], azimuth: 50, elevation: 22, width: 540, height: 380 });
  const { P } = f;
  const F = X([0, 0, 20]);
  f.part("housing", S(k.slabOf({ x: -40, y: -150, w: 90, d: 300, r: 20 }, -60, 160, P, 4, 1), { tone: "lo" }) + S(G.disc(50, 64, 56, F, P), { tone: "lo" }));
  f.part("bulb", S(G.disc(64, 70, 26, F, P), { tone: "hi" }) + S(G.disc(70, 78, 10, F, P), { tone: "mid" }), [140, 0, 0]);
  f.part("connector", S(G.prismOf(F, [[-14, -10], [14, -10], [14, 10], [-14, 10]], 78, 82, P), { tone: "lo" }), [200, 0, 0]);
  f.part("cover", S(G.disc(64, 84, 62, F, P), { tone: "mid" }) + L(G.circleOf(F, 84, 48, P, 48), { tone: "lo" }), [90, 0, 0]);
  void Z;
  return f.svg();
}

// Four ignition coils in a row on top of the cylinder head, the spark plugs in their wells below.
export function ignition() {
  const f = figure({ id: "ignition", label: "The top of a four-cylinder head drawn see-through: four ignition coils standing in their spark plug wells, and the spark plugs at the bottom of the wells.", fit: [[-200, -70, -60], [200, 70, -60], [-200, -70, 130], [200, 70, 130]], azimuth: 30, elevation: 26, width: 560, height: 420 });
  const { P } = f;
  const xs = [-135, -45, 45, 135];
  f.part("head", `<g class="rb-ghost">${S(k.slabOf({ x: -190, y: -65, w: 380, d: 130, r: 10 }, -60, 100, P, 4, 1), { tone: "lo" })}</g>` + L(xs.map((x) => k.ring(x, 0, 16, 40, P, 32)).join(""), { tone: "lo" }));
  const plug = (x) => S(k.cylinder(x, 0, 4, -52, 10, P, 12, 0.3), { tone: "mid" }) + S(k.cylinder(x, 0, 9, -42, 14, P, 6, 0.3), { tone: "lo" }) + S(k.cylinder(x, 0, 6, -28, 30, P, 16, 0.3), { tone: "hi" });
  f.part("plugs", xs.map(plug).join(""), [0, 0, 120]);
  const coil = (x) => S(k.cylinder(x, 0, 11, -2, 52, P, 24, 0.3), { tone: "mid" }) + S(k.slabOf({ x: x - 24, y: -26, w: 48, d: 52, r: 8 }, 50, 34, P, 3, 0.6), { tone: "hi" }) + S(k.slabOf({ x: x - 9, y: 26, w: 18, d: 14, r: 3 }, 62, 14, P, 2, 0.5), { tone: "lo" });
  f.part("coils", xs.map(coil).join(""), [0, 0, 160]);
  return f.svg();
}

// The auxiliary (ribbed) belt round its pulleys, seen from the end of the engine.
export function beltDrive() {
  const f = figure({ id: "belt", label: "The end of an engine with the auxiliary belt running round the crankshaft pulley, the alternator, the air conditioning compressor and the belt tensioner.", fit: [[0, -190, -100], [0, 230, -100], [0, -190, 280], [0, 230, 280], [60, 0, 0]], azimuth: 22, elevation: 14, width: 520, height: 440 });
  const { P } = f;
  const F = X([0, 0, 0]);
  const pulleys = [[0, 0, 74, "crank"], [-120, 220, 34, "alternator"], [150, 50, 56, "ac"], [70, 170, 32, "tensioner"]];
  const at = (y, z) => G.frameOf([0, y, z], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
  f.part("engine", `<g class="rb-ghost">${S(k.slabOf({ x: -120, y: -180, w: 100, d: 400, r: 10 }, -90, 360, P, 4, 1), { tone: "lo" })}</g>`);
  const disc = ([y, z, r]) => S(G.disc(-20, 22, r, at(y, z), P), { tone: "mid" }) + S(G.disc(22, 26, r * 0.45, at(y, z), P), { tone: "hi" });
  f.part("pulleys", pulleys.filter((p) => p[3] !== "tensioner").map(disc).join(""));
  f.part("tensioner", disc(pulleys[3]) + S(G.prismOf(at(pulleys[3][0], pulleys[3][1]), [[-8, -60], [8, -60], [8, 0], [-8, 0]], -24, -20, P), { tone: "lo" }), [0, 30, 30]);
  const ring = pulleys.flatMap(([y, z, r]) => Array.from({ length: 48 }, (_, i) => [y + (r + 3) * Math.cos((i / 48) * 2 * Math.PI), z + (r + 3) * Math.sin((i / 48) * 2 * Math.PI)]));
  const loop = G.hullOf(ring).map(([y, z]) => [18, y, z]);
  f.part("beltLoop", tubePieces(loop, 3, P, { closed: true, maxLength: 60, spacing: 1 }).map((piece) => tubeSvg(piece, { tone: "lo" })).join(""), [90, 0, 0]);
  void F; void Z;
  return f.svg();
}
