// EA888 Gen 2 (CAEB): compression and oil pressure tests, for oil consumption and oil pressure
// warnings. Values from the Audi repair manual, Rep. Gr. 15, p. 114–115 and Rep. Gr. 17, p. 171. PRIVATE.
import { manualMeta } from "../../lib/ea888.mjs";
import { ignition } from "../../lib/small-figures.mjs";
import { sump } from "../a4-b8-oil-change/figures.mjs";

export const meta = manualMeta({
  id: "ea888-compression-oil-pressure",
  title: "Oil consumption: compression and oil pressure tests",
  system: "Engine",
  commonProblem: "High oil consumption",
  minutes: 120,
  difficulty: "Intermediate",
  summary: "Early 2.0 TFSI engines are commonly reported to use oil through worn piston rings. Measure how much it uses, test compression on all four cylinders, and check oil pressure.",
});

const CYL = ["Cylinder 1", "Cylinder 2", "Cylinder 3", "Cylinder 4"];
const SPEC = {
  compression: { min: 159.5, max: 203.1, lowLimit: 101.5, unit: "psi", ref: "15 · p. 115" },
  oilIdle: { min: 17.4, max: 29, unit: "psi", ref: "17 · p. 171" },
  oil2000: { min: 23.2, max: 31.9, unit: "psi", ref: "17 · p. 171" },
  oil3700: { min: 43.5, max: 58, unit: "psi", ref: "17 · p. 171" },
  // How much oil it used, in quarts per 1,000 miles: recorded, compared with what Audi allows (not in this manual).
  consumption: { pending: true, unit: "qt per 1,000 mi", why: "compare with Audi's allowance for your car" },
};

const GROUPS = [
  { id: "measure", name: "Measure oil use" },
  { id: "compression", name: "Compression test" },
  { id: "pressure", name: "Oil pressure test" },
];

const STEPS = [
  {
    id: "consumption",
    group: "measure",
    title: "Measure how much oil it uses",
    figure: "sump",
    focus: ["pan"],
    ref: "",
    say: "Oil use is commonly reported on early CAEB engines, often from piston rings that stick or wear, or from a failed crankcase breather (see the PCV job). Measure it properly before deciding anything: fill to the top mark on level ground with the engine warm, note the mileage, drive normally for 1,000 miles, then measure how much it takes to reach the same mark.",
    tasks: [
      { type: "check", items: ["Level set exactly to the top mark, engine warm, car level; mileage noted", "Driven about 1,000 miles normally", "Oil needed to return to the same mark measured"] },
      { type: "measure", spec: "consumption", label: "Oil used", items: ["Per 1,000 miles"] },
      { type: "check", items: ["Blue smoke at start-up or under acceleration noted, if any", "Crankcase breather checked first (see the PCV job): a failed one also causes oil use"] },
    ],
  },
  {
    id: "compression-prep",
    group: "compression",
    title: "Prepare for the compression test",
    figure: "ignition",
    focus: ["coils", "plugs"],
    remove: ["coils", "plugs"],
    ref: "15 · p. 114",
    say: "The engine oil must be at least 86 °F (30 °C), and the battery at least 12.7 V, so the starter spins at full speed. Remove all four ignition coils and spark plugs.",
    tasks: [{ type: "check", items: ["Oil at least 86 °F (30 °C) · 15 · p. 114", "Battery at least 12.7 V; charger connected if needed · 15 · p. 114", "All four coils and spark plugs out (tool 3122B)", "Compression tester VAG1763 with adapters VAG1381/1 and /5A"] }],
  },
  {
    id: "compression",
    group: "compression",
    title: "Test each cylinder",
    figure: "ignition",
    focus: ["plugs"],
    ref: "15 · p. 115",
    say: "Fit the tester in each plug hole in turn. A helper cranks the engine until the reading stops rising. Write down each cylinder's reading. The difference between the highest and lowest cylinder must not be more than 43.5 psi (3.0 bar).",
    tasks: [
      { type: "measure", spec: "compression", label: "Compression pressure", items: CYL },
      { type: "check", items: ["Difference between highest and lowest cylinder no more than 43.5 psi (3.0 bar) · 15 · p. 115"] },
      {
        type: "choice",
        q: "One cylinder reads low. How do you tell rings from valves?",
        options: [
          { text: "Squirt a little engine oil into that cylinder and test again: if it rises clearly, the rings are worn", ok: true, why: "The oil seals the rings for a moment. No change points to the valves or head gasket. (A common workshop test, not from the manual.)" },
          { text: "Test it again without changing anything", ok: false, why: "Repeating confirms the reading but doesn't tell you the cause." },
          { text: "A low cylinder always means the head gasket", ok: false, why: "Rings, valves or the gasket can all cause it." },
        ],
      },
    ],
  },
  {
    id: "compression-close",
    group: "compression",
    title: "Refit plugs and coils, clear codes",
    figure: "ignition",
    focus: ["plugs", "coils"],
    install: ["plugs", "coils"],
    ref: "15 · p. 115, 28 · p. 253",
    say: "Refit the spark plugs and coils, then clear the fault codes the test set in the engine control module.",
    tasks: [
      { type: "torque", what: "Spark plugs", stages: [{ bolts: "All four", nm: 30 }] },
      { type: "check", items: ["Coils pushed fully on, connectors clicked in", "Fault codes from the test cleared with the diagnostic tool · 15 · p. 115"] },
    ],
  },
  {
    id: "oil-pressure",
    group: "pressure",
    title: "Test the oil pressure",
    figure: "sump",
    focus: ["filter"],
    ref: "17 · p. 171",
    say: "Low oil pressure starves the chain tensioner and bearings. With the oil level correct and the oil at least 176 °F (80 °C) (the radiator fan has run once), fit the gauge in place of the reduced oil pressure switch F378 on the oil filter bracket, refit the switch into the gauge, and read the pressure at three engine speeds.",
    tasks: [
      { type: "check", items: ["Oil level correct", "Oil at least 176 °F (80 °C): radiator fan has started once", "Gauge VAG1342 in place of switch F378, F378 fitted into the gauge · 17 · p. 171"] },
      { type: "measure", spec: "oilIdle", label: "Oil pressure at idle", items: ["Reading"] },
      { type: "measure", spec: "oil2000", label: "Oil pressure at 2,000 rpm", items: ["Reading"] },
      { type: "measure", spec: "oil3700", label: "Oil pressure at 3,700 rpm", items: ["Reading"] },
      { type: "note", text: "During break-in, the pressure at 2,000 rpm may read 43.5 to 58 psi (3.0 to 4.0 bar) (17 · p. 171)." },
      { type: "check", items: ["Switch F378 refitted to the bracket · 17 · p. 169", "No oil leaks at the switch"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: SPEC, diagrams: {}, figures: { ignition: ignition(), sump: sump() } };
}
