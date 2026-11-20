// The engine bay from above the front of the car: the engine with its oil filler cap and
// dipstick, the coolant expansion tank, the brake fluid reservoir and the washer fluid filler.
// A generic layout to show what each part looks like, not where it sits on a B8: use the
// symbols on the caps to find each one in your car.
import { figure, S, L, k } from "../../lib/figure.mjs";

export function bay() {
  const f = figure({ id: "bay", label: "An engine bay seen from above: the engine with its oil filler cap and dipstick, the coolant expansion tank, the brake fluid reservoir and the washer fluid filler cap.", fit: [[-300, -170, 0], [300, -170, 0], [-300, 170, 0], [300, 170, 0], [0, 0, 160]], azimuth: 30, elevation: 40, width: 600, height: 420 });
  const { P } = f;
  const items = [];
  const add = (name, centre, svg, apart) => items.push({ name, d: f.depth(centre), svg, apart });
  const slab = (x, y, w, d, z, h, tone, r = 8) => S(k.slabOf({ x, y, w, d, r }, z, h, P, 4, 1), { tone });
  const cyl = (x, y, r, z, h, tone) => S(k.cylinder(x, y, r, z, h, P, 32, 0.4), { tone });

  add("engine", [-20, 0, 40], slab(-150, -110, 260, 220, 0, 70, "lo", 18) + slab(-120, -80, 200, 160, 70, 12, "mid", 14) + L(k.pathOf([k.iso([-100, -60, 82.5], P), k.iso([60, -60, 82.5], P)]) + k.pathOf([k.iso([-100, 60, 82.5], P), k.iso([60, 60, 82.5], P)]), { tone: "lo" }));
  add("oilCap", [-40, 30, 90], cyl(-40, 30, 16, 82, 10, "hi") + cyl(-40, 30, 6, 92, 4, "mid"), [0, 0, 60]);
  add("dipstick", [40, -40, 95], cyl(40, -40, 4, 82, 16, "mid") + S(k.slabOf({ x: 31, y: -46, w: 18, d: 12, r: 4 }, 98, 5, P, 2, 0.5), { tone: "hi" }), [0, 0, 70]);
  add("coolantTank", [190, -110, 60], slab(150, -150, 90, 80, 20, 70, "hi", 12) + cyl(195, -110, 15, 90, 12, "mid"), [0, 0, 0]);
  add("brakeReservoir", [-240, -120, 70], slab(-275, -150, 70, 60, 40, 50, "hi", 10) + cyl(-240, -120, 13, 90, 10, "mid"), [0, 0, 0]);
  add("washerCap", [220, 120, 40], cyl(220, 120, 9, 0, 40, "lo") + cyl(220, 120, 14, 40, 8, "hi"), [0, 0, 50]);
  items.sort((a, b) => a.d - b.d).forEach((it) => f.part(it.name, it.svg, it.apart));
  return f.svg();
}
