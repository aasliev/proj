// Renderers for the parts that turn with the crank. `makeMoving` takes the kit
// (k), the lathe module (G), the projection (P) and the model (M) as arguments
// and has no imports, so the build and the page run the same source.
//
// Each renderer returns parts in painter's order: { id, at, sig, svg() }. `sig`
// changes only when that part looks different, so the page redraws just those
// parts; explode offsets are a transform on the whole group.
export function makeMoving(k, G, P, M) {
  const DEG = Math.PI / 180;
  const V = G.viewOf(P);
  const depth = (p) => G.dot3(p, V);
  const S = (paths, style) => k.solidSvg(paths, style);
  const L = (d, style) => k.lineSvg(d, style);
  const D = (points, style) => k.dotsSvg(points, style);
  const X = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
  const disc = (s0, s1, r, F) => G.disc(s0, s1, r, F, P, { steps: r >= 30 ? 48 : r >= 15 ? 32 : 20 });
  const yz = (deg, r) => [r * Math.sin(deg * DEG), r * Math.cos(deg * DEG)];
  const hull = (points) => {
    const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], hi = [];
    for (const p of pts) { while (lo.length > 1 && cross(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
    for (const p of pts.reverse()) { while (hi.length > 1 && cross(hi[hi.length - 2], hi[hi.length - 1], p) <= 0) hi.pop(); hi.push(p); }
    return lo.slice(0, -1).concat(hi.slice(0, -1));
  };
  const circle = (cy, cz, r, n = 20) => Array.from({ length: n }, (_, i) => [cy + r * Math.cos((i / n) * 2 * Math.PI), cz + r * Math.sin((i / n) * 2 * Math.PI)]);
  const deg = (value, step = 0.5) => (Math.round(value / step) * step).toFixed(1);
  // Collect parts; sort once by depth.
  function collector() {
    const parts = [];
    const put = (id, centre, sig, svg, at) => parts.push({ id, at: at ?? depth(centre), sig, svg });
    put.done = () => parts.sort((a, b) => a.at - b.at);
    return put;
  }

  const MAINS = [-176, -88, 0, 88, 176];
  const W = 11; // half width of a journal or pin
  // Sprocket planes along x and radii: [x, r, teeth].
  const SPROCKETS = {
    crank: [[197, 21, 18], [207, 24, 20], [216, 19, 16]],
    balance: { at: [[70, 22], [-70, -22]], x: 197, r: 24, teeth: 20 },
    cam: { x: 207, r: 44, teeth: 36 },
    pump: { at: [0, -150], x: 216, r: 26, teeth: 22 },
  };
  // Teeth as dots on the visible face of a sprocket, turned by `turn` degrees (frame angle).
  const teethOf = (F, s, r, count, turn) => {
    const step = (2 * Math.PI) / count;
    return G.dotsOf(F, s, r, count, P, { all: true, phase: (((turn * DEG) % step) + step) % step });
  };
  const toothSig = (turn, count) => deg((((turn % (360 / count)) + 360 / count) % (360 / count)), 1);

  // A crank web: hub round the main, round the pin, and a counterweight opposite the pin.
  function webPolygon(theta, cyl) {
    const a = theta + M.throwDeg[cyl];
    const [py, pz] = yz(a, M.r);
    const weight = [];
    for (let t = -58; t <= 58; t += 8) weight.push(yz(a + 180 + t, 74));
    return hull([...circle(0, 0, 31), ...circle(py, pz, 29), ...weight]);
  }

  function crank(theta) {
    const put = collector();
    const th = deg(theta);
    for (const x of MAINS) {
      put(`main${x}`, [x, 0, 0], "s", () => S(disc(-W, W, M.mainR, X([x, 0, 0])), { tone: "hi" }) + L(G.arcOf(X([x, 0, 0]), -W + 2.5, M.mainR, P), { tone: "faint" }));
      // Main bearing cap under the journal, bolted from below.
      put(`cap${x}`, [x, 0, -M.mainR - 11], "s", () => S(k.slabOf({ x: x - W, y: -42, w: 2 * W, d: 84, r: 2 }, -M.mainR - 22, 20, P, 3, 0.5), { tone: "mid" }) + D([[x, -32], [x, 32]].map(([cx, cy]) => k.iso([cx, cy, -M.mainR - 22], P)), { size: 0.5, tone: "mid" }));
    }
    for (const cyl of [1, 2, 3, 4]) {
      const cx = M.cylX[cyl];
      const [py, pz] = yz(theta + M.throwDeg[cyl], M.r);
      for (const [s0, s1] of [[cx - W - 22, cx - W], [cx + W, cx + W + 22]]) {
        put(`web${s0}`, [(s0 + s1) / 2, 0, 0], th, () => S(G.prismOf(X([0, 0, 0]), webPolygon(theta, cyl), s0, s1, P), { tone: "mid" }));
      }
      put(`pin${cyl}`, [cx, 0, 0.1], th, () => S(disc(-W, W, M.pinR, X([cx, py, pz])), { tone: "hi" }));
    }
    // Nose with its three sprockets (balance, cam, oil pump), then the rear flange.
    put("nose", [215, 0, 0], "s", () => S(disc(0, 54, 19, X([187, 0, 0])), { tone: "hi" }) + L(G.circleOf(X([187, 0, 0]), 54, 6, P, 16), { tone: "lo" }));
    for (const [x, r, teeth] of SPROCKETS.crank) {
      put(`spr${x}`, [x, 0, 0.5], toothSig(-theta, teeth), () => S(disc(-3.5, 3.5, r, X([x, 0, 0])), { tone: "mid" }) + D(teethOf(X([x, 0, 0]), 3.5, r - 1.6, teeth, -theta), { size: 0.45, tone: "lo" }));
    }
    put("flange", [-195, 0, 0], "s", () => S(disc(0, 13, 42, X([-200, 0, 0])), { tone: "mid" }));
    return put.done();
  }

  // A connecting rod in the y–z plane, 20 thick, and its piston.
  function pistons(theta) {
    const put = collector();
    for (const cyl of [1, 2, 3, 4]) {
      const cx = M.cylX[cyl];
      const [py, pz] = yz(theta + M.throwDeg[cyl], M.r);
      const pin = M.pistonPin(theta, cyl);
      const sig = deg(theta);
      const base = depth([cx, 0, 100]);
      put(`rod${cyl}`, null, sig, () => {
        const rod = hull([...circle(py, pz, 31, 28), ...circle(0, pin, 13, 16)]);
        const lean = Math.atan2(py, pin - pz);
        // The cap joint across the big end, square to the rod.
        const across = [Math.cos(lean), -Math.sin(lean)];
        const face = cx + 10;
        const seam = k.pathOf([k.iso([face, py - across[0] * 31.4, pz - across[1] * 31.4], P), k.iso([face, py + across[0] * 31.4, pz + across[1] * 31.4], P)]);
        const bolts = [-22, 22].map((t) => k.iso([face, py + across[0] * t - Math.sin(lean) * 6, pz + across[1] * t - Math.cos(lean) * 6], P));
        return S(G.prismOf(X([0, 0, 0]), rod, cx - 10, cx + 10, P), { tone: "mid", inner: L(seam, { tone: "lo" }) }) + L(G.circleOf(X([cx, py, pz]), 10, 24, P, 28), { tone: "lo" }) + D(bolts, { size: 0.6, tone: "mid" });
      }, base - 0.01);
      // Piston: crown with valve reliefs, three ring grooves, the pin boss.
      put(`piston${cyl}`, null, sig, () => {
        const r = M.bore / 2 - 0.25;
        const top = pin + M.compressionHeight;
        const rings = [4, 7.5, 11].map((d) => k.sideArc(cx, 0, r, top - d, P, 40)).join("");
        const reliefs = [[-14, -18], [14, -18], [-14, 18], [14, 18]].map(([dx, dy]) => k.ring(cx + dx, dy, 11, top, P, 24)).join("");
        return S(k.cylinder(cx, 0, r, pin - M.skirt, top - pin + M.skirt, P, 64, 0.8), { tone: "hi", crease: "lo", inner: L(rings, { tone: "lo" }) }) + L(reliefs, { tone: "faint" }) + L(k.sideRing([cx, r, pin], 9.5, "y", P, 24), { tone: "lo" }) + D([k.iso([cx, r, pin], P)], { size: 0.6, tone: "mid" });
      }, base);
    }
    return put.done();
  }

  // ---- Balance shafts: twice crank speed, one each way, eccentric weights.
  function balance(theta) {
    const put = collector();
    SPROCKETS.balance.at.forEach(([y, z], index) => {
      const turn = index === 0 ? 2 * theta : -2 * theta;
      const F = X([0, y, z]);
      const cuts = [-190, -160, -140, -120, -40, -10, 10, 40, 120, 140, 160, 190];
      const kind = (a, b) => ((a >= -120 && b <= -40) || (a >= 40 && b <= 120) ? "weight" : (a >= -160 && b <= -140) || (a >= -10 && b <= 10) || (a >= 140 && b <= 160) ? "journal" : "shaft");
      for (let i = 0; i < cuts.length - 1; i++) {
        const [a, b] = [cuts[i], cuts[i + 1]];
        const what = kind(a, b);
        if (what === "weight") {
          put(`w${index}${a}`, [(a + b) / 2, y, z], deg(turn), () => {
            const half = [[0, 0]];
            for (let t = -90; t <= 90; t += 10) half.push(yz(turn + t, 30));
            return S(disc(a, b, 10, F), { tone: "hi" }) + S(G.prismOf(F, hull(half), a, b, P), { tone: "mid" });
          });
        } else put(`s${index}${a}`, [(a + b) / 2, y, z], "s", () => S(disc(a, b, what === "journal" ? 15 : 10, F), { tone: what === "journal" ? "hi" : "mid" }));
      }
      const Fs = X([SPROCKETS.balance.x, y, z]);
      put(`n${index}`, [192, y, z], "s", () => S(disc(190, 194, 10, F), { tone: "mid" }));
      put(`spr${index}`, [SPROCKETS.balance.x, y, z], toothSig(-turn, SPROCKETS.balance.teeth), () => S(disc(-3.5, 3.5, SPROCKETS.balance.r, Fs), { tone: "mid" }) + D(teethOf(Fs, 3.5, SPROCKETS.balance.r - 1.6, SPROCKETS.balance.teeth, -turn), { size: 0.45, tone: "lo" }));
    });
    return put.done();
  }

  // ---- Camshafts: half crank speed. Intake (-y) carries the vane adjuster and the
  // four-lobe pump cam; exhaust (+y) carries the Audi valvelift sleeves.
  const VALVE_DX = 16;
  function lobe(cy, cz, gamma, lift, s0, s1) {
    const nose = 19 + lift - 7;
    const poly = hull([...circle(cy, cz, 19, 24), ...circle(cy + nose * Math.sin(gamma * DEG), cz + nose * Math.cos(gamma * DEG), 7, 12)]);
    return S(G.prismOf(X([0, 0, 0]), poly, s0, s1, P), { tone: "hi" });
  }
  function cams(theta) {
    const put = collector();
    const cam = M.camDeg(theta);
    for (const kind of ["intake", "exhaust"]) {
      const cy = M.camY[kind], cz = M.camZ;
      const F = X([0, cy, cz]);
      const features = [];
      for (const j of MAINS) features.push([j - 9, j + 9, "journal"]);
      for (const cyl of [1, 2, 3, 4]) {
        const cx = M.cylX[cyl];
        if (kind === "exhaust") features.push([cx - 27, cx + 27, "sleeve", cyl]);
        else for (const dx of [-VALVE_DX, VALVE_DX]) features.push([cx + dx - 7, cx + dx + 7, "lobe", cyl]);
      }
      features.sort((a, b) => a[0] - b[0]);
      let from = -195;
      for (const [a, b, what, cyl] of features) {
        const f0 = from;
        if (a > f0) put(`${kind}s${a}`, [(f0 + a) / 2, cy, cz], "s", () => S(disc(f0, a, 12, F), { tone: "mid" }));
        const gamma = cyl ? 180 + M.lobeDeg(theta, cyl, kind) : 0;
        const g = deg(gamma);
        if (what === "journal") put(`${kind}j${a}`, [(a + b) / 2, cy, cz], "s", () => S(disc(a, b, 16, F), { tone: "hi" }));
        else if (what === "lobe") put(`${kind}l${a}`, [(a + b) / 2, cy, cz], g, () => lobe(cy, cz, gamma, M.valveLift[kind] / 2, a, b));
        else {
          // Sliding cam piece: a sleeve with two lobe pairs and a spiral groove.
          const cx = M.cylX[cyl];
          put(`${kind}a${cx}`, [cx - 26, cy, cz], "s", () => S(disc(a, a + 6, 17, F), { tone: "mid" }) + L(G.arcOf(F, a + 3, 17, P), { tone: "lo" }));
          for (const dx of [-VALVE_DX, VALVE_DX]) {
            put(`${kind}p${cx + dx}`, [cx + dx - 3.5, cy, cz], g, () => lobe(cy, cz, gamma, 6.35 / 2, cx + dx - 10, cx + dx - 3));
            put(`${kind}q${cx + dx}`, [cx + dx + 3.5, cy, cz], g, () => lobe(cy, cz, gamma, M.valveLift.exhaust / 2, cx + dx - 3, cx + dx + 7));
          }
          put(`${kind}g${cx}`, [cx, cy, cz], deg(cam, 2), () => S(disc(cx - 9, cx + 9, 17, F), { tone: "mid" }) + L(G.ribsOf(F, cx - 9, cx + 9, 17, 6, P, { twist: 0.6, phase: -cam * DEG }), { tone: "lo" }));
          put(`${kind}b${cx}`, [b - 3, cy, cz], "s", () => S(disc(b - 6, b, 17, F), { tone: "mid" }));
        }
        from = b;
      }
      const f1 = from;
      put(`${kind}end`, [(f1 + 196) / 2, cy, cz], "s", () => S(disc(f1, 196, 12, F), { tone: "mid" }));
      const Fs = X([SPROCKETS.cam.x, cy, cz]);
      if (kind === "intake") {
        // Vane adjuster: housing with the sprocket, front cover and bolts.
        put(`${kind}adj`, [209, cy, cz], toothSig(-cam, 4), () => S(disc(-11, 15, SPROCKETS.cam.r - 4, Fs), { tone: "hi" }) + L(G.circleOf(Fs, 15, 20, P, 32), { tone: "lo" }) + D(teethOf(Fs, 15, 30, 4, -cam), { size: 0.6 }));
      } else {
        put(`${kind}hub`, [209, cy, cz], toothSig(-cam, 3), () => S(disc(-4, 10, 18, Fs), { tone: "mid" }) + D(teethOf(Fs, 10, 11, 3, -cam), { size: 0.5 }));
      }
      put(`${kind}spr`, [207, cy, cz - 0.5], toothSig(-cam, SPROCKETS.cam.teeth), () => S(disc(-3.5, 3.5, SPROCKETS.cam.r, Fs), { tone: "mid" }) + D(teethOf(Fs, 3.5, SPROCKETS.cam.r - 1.6, SPROCKETS.cam.teeth, -cam), { size: 0.45, tone: "lo" }));
      if (kind === "intake") {
        // Four-lobe cam for the high-pressure fuel pump at the rear.
        put(`${kind}pump`, [-201, cy, cz], toothSig(cam, 4), () => {
          const four = [];
          for (let t = 0; t < 360; t += 6) {
            const r = 17 + 3.5 * Math.cos(4 * (t + cam) * DEG);
            four.push([cy + r * Math.sin(t * DEG), cz + r * Math.cos(t * DEG)]);
          }
          return S(G.prismOf(X([0, 0, 0]), hull(four), -207, -195, P), { tone: "hi" });
        });
      }
    }
    return put.done();
  }

  // ---- Valves, springs and roller finger followers, moved by the cam lift.
  function valves(theta) {
    const put = collector();
    for (const kind of ["intake", "exhaust"]) {
      const sign = kind === "intake" ? -1 : 1;
      for (const cyl of [1, 2, 3, 4]) {
        const lift = M.lift(theta, cyl, kind);
        const sig = lift.toFixed(1);
        for (const dx of [-VALVE_DX, VALVE_DX]) {
          const x = M.cylX[cyl] + dx;
          const y = sign * 30;
          const top = 304 - lift;
          put(`v${kind}${x}`, [x, y, 290], sig, () => {
            const coils = [];
            for (let i = 0; i <= 7; i++) coils.push(k.sideArc(x, y, 11, 268 + ((top - 3 - 268) * i) / 7, P, 20));
            // The coils end on the spring's two limb lines.
            const side = [-V[1], V[0]];
            const n = Math.hypot(side[0], side[1]);
            const limbs = [1, -1].map((sg) => k.segment([x + (sg * 11 * side[0]) / n, y + (sg * 11 * side[1]) / n, 268], [x + (sg * 11 * side[0]) / n, y + (sg * 11 * side[1]) / n, top - 3], P)).join("");
            return L(coils.join("") + limbs, { tone: "lo" }) + S(k.cylinder(x, y, 12, top - 3, 3, P, 24, 0.4), { tone: "mid" }) + S(k.cylinder(x, y, 3.5, top, 4, P, 12), { tone: "hi" });
          });
          // The roller sits midway, so it drops half the valve lift; the lash adjuster end stays put.
          put(`f${kind}${x}`, [x, y + sign * 18, 312], sig, () => {
            const roller = 310 - lift / 2;
            const finger = { x: x - 6, y: sign > 0 ? y - 4 : y - 36, w: 12, d: 40, r: 3 };
            return S(k.cylinder(x, sign * 62, 6, 290, 14, P, 16), { tone: "mid" }) + S(k.slabOf(finger, roller - 6, 6, P, 3, 0.4), { tone: "mid" }) + S(disc(-5, 5, 6, X([x, sign * 54, roller])), { tone: "hi" });
          });
        }
      }
    }
    return put.done();
  }

  // Sprockets, the adjuster and the crank nose stand proud of the block and head, in
  // front of them, so each turning assembly is drawn as a body and a separate front.
  const FRONT = /^(nose|spr\d|n\d|spr\d|intakeadj|exhausthub|intakespr|exhaustspr)/;
  const isFront = (part) => FRONT.test(part.id);
  const body = (fn) => (theta) => fn(theta).filter((part) => !isFront(part));
  const front = (fn) => (theta) => fn(theta).filter(isFront);
  const markup = (parts) => parts.map((p) => `<g data-p="${p.id}">${p.svg()}</g>`).join("");
  return {
    crank: body(crank),
    crankFront: front(crank),
    balance: body(balance),
    balanceFront: front(balance),
    cams: body(cams),
    camsFront: front(cams),
    pistons,
    valves,
    markup, webPolygon, hull, circle, MAINS, SPROCKETS, VALVE_DX };
}
