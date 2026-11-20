// Shared drawing helpers for job figures, on top of the Anatomy kit in .claude/skills/anatomy.
// Each figure is one SVG; every part sits in <g data-part="id" data-apart="dx,dy"> so the job
// page can hold it apart (shifted by that screen offset) until its step is done.
import * as k from "../../.claude/skills/anatomy/kit/iso-kit.mjs";
import * as G from "../../.claude/skills/anatomy/kit/lathe.mjs";

export { k, G };
export const DEG = Math.PI / 180;
export const S = (paths, style) => k.solidSvg(paths, style);
export const L = (d, style) => k.lineSvg(d, style);
export const D = (points, style) => k.dotsSvg(points, style);
export const X = (o) => G.frameOf(o, [1, 0, 0], [0, 1, 0], [0, 0, 1]);
export const Y = (o) => G.frameOf(o, [0, 1, 0], [0, 0, 1], [1, 0, 0]);
export const Z = (o) => G.frameOf(o, [0, 0, 1], [1, 0, 0], [0, 1, 0]);

// A figure: fit a camera to the points in `fit`, then add parts far to near.
export function figure({ id, width = 560, height = 400, fit, azimuth = 40, elevation = 28, pad = 26, label }) {
  const P = k.fitProjection(fit, width, height, { pad, azimuth, elevation });
  const V = G.viewOf(P);
  const at = (p) => k.iso(p, P);
  const parts = [];
  return {
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
    // Several [depth, svg] solids, far to near.
    sorted: (list) => list.sort((a, b) => a[0] - b[0]).map((entry) => entry[1]).join(""),
    svg: () => `<svg xmlns="http://www.w3.org/2000/svg" class="iso-svg rb-fig" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}" data-fig="${id}">${parts.join("")}</svg>`,
  };
}

// An annular sector (a pad, a bracket, a caliper bridge) as a polygon in a frame's (u, v) plane.
export function sector(r0, r1, a0, a1, steps = 12) {
  const out = [];
  for (let i = 0; i <= steps; i++) {
    const t = (a0 + ((a1 - a0) * i) / steps) * DEG;
    out.push([r1 * Math.cos(t), r1 * Math.sin(t)]);
  }
  for (let i = steps; i >= 0; i--) {
    const t = (a0 + ((a1 - a0) * i) / steps) * DEG;
    out.push([r0 * Math.cos(t), r0 * Math.sin(t)]);
  }
  return out;
}
