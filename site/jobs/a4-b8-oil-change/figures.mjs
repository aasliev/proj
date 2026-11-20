// The oil pan with its drain plug and seal, the oil filter above it, and a drain pan below.
// A generic layout, not measured from the CAEB: it shows the parts and the order they come off.
import { figure, S, L, X, G, k } from "../../lib/figure.mjs";

export function sump() {
  const f = figure({ id: "sump", label: "The bottom of an engine: the oil pan with the drain plug and its seal ring on the side facing you, the oil filter above, and a drain pan on the floor below.", fit: [[-150, -110, -130], [260, -110, -130], [-150, 110, 220], [260, 110, 220]], azimuth: 52, elevation: 18, width: 560, height: 440 });
  const { P } = f;
  const F = X([130, 0, 22]);
  f.part("filter", S(k.cylinder(60, -150, 38, 130, 90, P, 48, 0.5), { tone: "hi" }) + L(k.ring(60, -150, 30, 220, P, 48), { tone: "lo" }), [0, 0, 110]);
  f.part("catchPan", S(k.cylinder(190, 0, 80, -130, 28, P, 64, 0.5), { tone: "lo" }) + L(k.ring(190, 0, 72, -102, P, 64), { tone: "lo" }));
  f.part("block", `<g class="rb-ghost">${S(k.slabOf({ x: -130, y: -100, w: 260, d: 200, r: 10 }, 90, 120, P, 4, 1), { tone: "lo" })}</g>`);
  f.part("pan", S(k.slabOf({ x: -130, y: -100, w: 260, d: 200, r: 14 }, 0, 90, P, 4, 1), { tone: "mid" }) + S(G.disc(-2, 0, 16, F, P), { tone: "lo" }));
  f.part("sealRing", S(G.ringBand(0, 3, 9, 15, F, P), { tone: "hi" }), [60, 0, 0]);
  f.part("drainPlug", S(G.disc(0, 14, 8, F, P), { tone: "mid" }) + S(G.disc(14, 24, 14, F, P), { tone: "hi" }), [110, 0, -30]);
  return f.svg();
}
