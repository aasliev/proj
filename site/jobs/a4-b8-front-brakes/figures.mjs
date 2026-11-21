// One front brake corner, seen from outboard: wheel hub, brake disc, caliper carrier,
// the two pads, the caliper and the wheel. The axle runs along x; +x is outboard.
// Proportions are generic, not measured from a B8 part.
import { figure, sector, S, L, D, X, G, k } from "../../lib/figure.mjs";

export function corner(label = "A front brake corner seen from outboard: the hub, the brake disc, the caliper carrier over the top of the disc, the inner and outer pads, the caliper and the wheel.") {
  const f = figure({ id: "corner", label, fit: [[-30, -180, -170], [-30, 180, 260], [150, -180, -170], [150, 180, 260]], azimuth: 48, elevation: 20, width: 560, height: 460 });
  const { P } = f;
  const F = X([0, 0, 0]);
  const prism = (poly, s0, s1, style) => S(G.prismOf(F, poly, s0, s1, P), style);

  // Hub: flange and centring spigot, five bolt holes.
  f.part("hub", S(G.disc(-24, 10, 72, F, P), { tone: "lo" }) + S(G.disc(10, 22, 30, F, P), { tone: "mid" }) + D(G.dotsOf(F, 10, 52, 5, P, { all: true }), { size: 1.1, tone: "lo" }));
  // Inner pad sits behind the disc: drawn before it.
  const pad = (s0, s1) => prism(sector(108, 150, 64, 116), s0, s0 + 4, { tone: "mid" }) + prism(sector(112, 146, 67, 113), s0 + 4, s1, { tone: "lo" });
  f.part("innerPad", pad(-2, 10), [0, 0, 80]);
  // Disc: friction ring and hat, bolt holes in the hat face.
  const disc = S(G.ringBand(12, 40, 80, 157, F, P), { tone: "hi" }) + S(G.disc(12, 48, 82, F, P), { tone: "hi" }) + L(G.circleOf(F, 40, 120, P, 64), { tone: "faint", className: "rb-edge" }) + D(G.dotsOf(F, 48, 52, 5, P, { all: true }), { size: 1.2, tone: "lo" }) + S(G.disc(48, 49, 22, F, P), { tone: "lo" });
  f.part("disc", disc, [110, 0, 0]);
  // Carrier: a bridge over the disc's top edge with two lugs on the inboard side.
  f.part("carrier", prism(sector(150, 168, 58, 122), -10, 58, { tone: "lo" }) + prism(sector(96, 168, 52, 62), -18, -6, { tone: "lo" }) + prism(sector(96, 168, 118, 128), -18, -6, { tone: "lo" }), [0, 0, 110]);
  // Outer pad, in front of the disc.
  const outer = prism(sector(112, 146, 67, 113), 42, 50, { tone: "lo" }) + prism(sector(108, 150, 64, 116), 50, 54, { tone: "mid" });
  f.part("outerPad", outer, [0, 0, 80]);
  // Caliper: piston housing inboard, bridge over the top, fingers outboard.
  const caliper = prism(sector(102, 178, 66, 114), -30, -4, { tone: "mid" }) + prism(sector(160, 184, 66, 114), -30, 64, { tone: "mid" }) + prism(sector(116, 184, 72, 108), 54, 64, { tone: "hi" });
  f.part("caliper", caliper, [0, 0, 150]);
  // Wheel bolts.
  const bolts = [0, 1, 2, 3, 4].map((i) => {
    const t = (90 + i * 72) * Math.PI / 180;
    const B = X([0, 52 * Math.cos(t), 52 * Math.sin(t)]);
    return S(G.disc(60, 82, 9, B, P), { tone: "hi" }) + S(G.disc(82, 88, 12, B, P), { tone: "mid" });
  }).join("");
  // Wheel as a ghost so the brake stays visible through it.
  const wheel = `<g class="rb-ghost">${S(G.ringBand(60, 150, 172, 196, F, P), { tone: "lo" })}</g>` + L(G.circleOf(F, 150, 172, P, 72) + G.circleOf(F, 66, 70, P, 48) + G.spokesOf(F, 66, 70, 172, 5, P, { phase: Math.PI / 5 }), { tone: "lo" });
  f.part("wheel", wheel, [150, 0, 0]);
  f.part("bolts", bolts, [200, 0, 0]);
  void k;
  return f.svg();
}

// A pad and disc on the bench, for the measuring steps.
function bench() {
  const f = figure({ id: "bench", label: "A brake disc lying on the bench with a micrometer position marked on its friction ring, and a pad beside it with the friction material thickness to check.", fit: [[-170, -170, 0], [170, 170, 0], [-170, -170, 60], [170, 170, 60], [260, 220, 0]], azimuth: 36, elevation: 34, width: 560, height: 400 });
  const { P } = f;
  const F = G.frameOf([0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]);
  const disc = S(G.ringBand(0, 28, 80, 157, F, P), { tone: "hi" }) + S(G.disc(0, 36, 82, F, P), { tone: "hi" }) + D(G.dotsOf(F, 36, 52, 5, P, { all: true }), { size: 1.2, tone: "lo" }) + L(G.circleOf(F, 28, 140, P, 64), { tone: "lo", className: "rb-plane" });
  f.part("disc", disc);
  const Fp = G.frameOf([190, 170, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]);
  f.part("pads", S(G.prismOf(Fp, [[-70, -24], [70, -24], [62, 24], [-62, 24]], 0, 6, P), { tone: "mid" }) + S(G.prismOf(Fp, [[-64, -20], [64, -20], [56, 20], [-56, 20]], 6, 20, P), { tone: "lo" }));
  return f.svg();
}

export const buildFigures = () => ({ corner: corner(), bench: bench() });
