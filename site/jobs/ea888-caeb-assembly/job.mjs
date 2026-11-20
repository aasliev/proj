// EA888 Gen 2 (CAEB) engine assembly: the rebuild trainer's steps, drawn from the Audi factory
// repair manual. The data lives with the trainer in docs/figures/ea888-caeb/rebuild; this file
// adapts it to the job format. PRIVATE: built from a copyrighted manual, left out of public builds.
import { STEPS, GROUPS, VARIANTS, SPEC, DIAGRAMS, specFor } from "../../../docs/figures/ea888-caeb/rebuild/steps.mjs";
import { buildFigures } from "../../../docs/figures/ea888-caeb/rebuild/figures.mjs";

export const meta = {
  id: "ea888-caeb-assembly",
  title: "Engine assembly: bottom end, head, timing and pans",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2011" },
  engine: "EA888 Gen 2 · CAEB (2.0 TFSI)",
  system: "Engine",
  vehicleLine: "EA888 Gen 2 · CAEB (2.0 TFSI), 2011 Audi A4",
  status: "ready",
  private: true,
  minutes: 1800,
  difficulty: "Expert",
  summary: "Assemble the engine from the bare block: balance shafts, crank, pistons, head and camshafts, timing chains, covers, oil pans and coolant pump.",
  source: { name: "Audi repair manual, EA888 Gen II (Edition 06.2019)", prefix: "Repair manual, Rep. Gr. " },
  storageKey: "ea888-rebuild-v1",
  legal: "Practice tool built from the Audi factory repair manual for the EA888 Gen II engine (Edition 06.2019). Every value names its repair group and page: check it against your copy before you turn a bolt. Progress and measurements are saved in this browser only; export them to keep a copy. Not affiliated with Audi.",
};

// Ring gap and groove specs depend on the oil ring version: express them as a table the
// engine can pick from.
function ringSpecs() {
  const out = {};
  for (const id of ["gap1", "gap2", "gapOil", "groove1", "groove2"]) {
    const two = specFor(id, { oilRing: "2" });
    const three = specFor(id, { oilRing: "3" });
    const strip = ({ unit, ref, ...rest }) => rest;
    out[id] = { unit: two.unit, ref: two.ref, byVariant: { key: "oilRing", values: { 2: strip(two), 3: strip(three) } } };
  }
  return out;
}

export function build() {
  const diagrams = { ...DIAGRAMS, mains: { ...DIAGRAMS.mains, label: "bearing cap 1 / 5: check orientation in your manual, p. 38", hint: "Work from the middle cap outward." } };
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: { ...SPEC, ...ringSpecs() }, diagrams, figures: buildFigures() };
}
