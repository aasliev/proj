// EA888 Gen 2 (CAEB): checking the valve timing, to tell whether the timing chain has jumped
// or stretched. Values from the Audi repair manual, Rep. Gr. 15, p. 101–103. PRIVATE.
import { manualMeta } from "../../lib/ea888.mjs";
import { DIAGRAMS } from "../../../docs/figures/ea888-caeb/rebuild/steps.mjs";
import { buildFigures } from "../../../docs/figures/ea888-caeb/rebuild/figures.mjs";
import { ignition } from "../../lib/small-figures.mjs";

export const meta = manualMeta({
  id: "ea888-timing-chain-check",
  title: "Timing chain check (valve timing)",
  system: "Engine",
  commonProblem: "Timing chain stretch and tensioner wear",
  minutes: 120,
  difficulty: "Intermediate",
  summary: "A rattle at cold start or a cam/crank correlation fault code can mean the timing chain has stretched or jumped. Measure the camshaft marks at TDC against the manual's figures.",
});

const SPEC = {
  intakeMark: { min: 61, max: 64, unit: "mm", ref: "15 · p. 102" },
  camToCam: { min: 124, max: 126, unit: "mm", ref: "15 · p. 103" },
};

const GROUPS = [
  { id: "why", name: "Symptoms" },
  { id: "prep", name: "Preparation" },
  { id: "measure", name: "Measure" },
  { id: "close", name: "Results and reassembly" },
];

const STEPS = [
  {
    id: "symptoms",
    group: "why",
    title: "When to check the timing chain",
    figure: "front",
    focus: ["camChain", "camTensioner"],
    ref: "",
    say: "Owners and workshops commonly report two signs on this engine: a rattle from the timing side for a few seconds after a cold start, and fault codes for camshaft and crankshaft position correlation (for example P0016). Early chain tensioners were revised; a dealer can tell you from the VIN and the part whether yours is the updated one.",
    tasks: [
      { type: "check", items: ["Fault codes read with a diagnostic tool and noted", "Cold-start rattle listened for (from the timing chain side, first seconds after starting)", "Service history checked for a previous chain or tensioner replacement"] },
      {
        type: "choice",
        q: "The engine rattles for two seconds after a cold start, and there is a cam/crank correlation code. What next?",
        options: [
          { text: "Clear the code and keep driving", ok: false, why: "A chain that has jumped a tooth can let the valves hit the pistons. Check before driving more." },
          { text: "Check the valve timing as in this job, then decide", ok: true, why: "The measurement shows whether the cams are where they should be relative to the crank." },
          { text: "Change the oil to a thicker grade", ok: false, why: "That can hide the rattle, not fix the chain or tensioner." },
        ],
      },
    ],
  },
  {
    id: "tools",
    group: "prep",
    title: "Tools, and open up the timing side",
    figure: "front",
    focus: ["upperCover"],
    remove: ["upperCover"],
    ref: "15 · p. 101",
    say: "You need a dial gauge with a spark plug adapter to find TDC exactly. Remove the upper timing chain cover and the front noise insulation (the splash shield).",
    tasks: [{ type: "check", items: ["Spark plug tool 3122B, dial gauge VAS6341 with adapter T10170A · 15 · p. 101", "Wrench T40263 (21 mm) with adapter T40314 and a 24 mm socket, to turn the crank on the vibration damper · 15 · p. 101", "Upper timing chain cover removed · 15 · p. 68", "Front noise insulation removed (body manual, Rep. Gr. 66)"] }],
  },
  {
    id: "tdc",
    group: "prep",
    title: "Find TDC on cylinder 1 with the dial gauge",
    figure: "ignition",
    focus: ["coils", "plugs"],
    remove: ["coils", "plugs"],
    ref: "15 · p. 102",
    say: "Turn the crank in the direction of engine rotation until the marks are almost at the top. Take out cylinder 1's coil and spark plug, screw the dial gauge adapter all the way into the plug thread and lock the gauge in it. Turn the crank slowly forward until the gauge reaches its maximum: that is TDC.",
    tasks: [
      { type: "check", items: ["Crank turned only in the direction of engine rotation", "Coil and spark plug out of cylinder 1 · 28 · p. 254", "Adapter screwed fully into the plug thread, gauge inserted and locked", "Gauge at its maximum reading: piston at TDC"] },
      {
        type: "choice",
        q: "You turned the crank past the gauge's maximum. What does the manual say?",
        options: [
          { text: "Turn it back a little", ok: false, why: "Never turn the engine backward: the chain slack moves and the reading is wrong." },
          { text: "Turn it two more full turns forward and approach TDC again", ok: true, why: "Always come up to TDC in the direction of rotation (15 · p. 102)." },
          { text: "Carry on: a few degrees don't matter", ok: false, why: "The cam marks are measured in millimeters; a few degrees change the reading." },
        ],
      },
    ],
  },
  {
    id: "measure",
    group: "measure",
    title: "Measure the camshaft marks",
    figure: "front",
    focus: ["sprocket3", "camChain"],
    ref: "15 · p. 102–103",
    say: "First measure from the left outer edge of the cylinder head's ridge to the mark on the intake camshaft. When that is in spec, measure between the mark on the exhaust camshaft and the mark on the intake camshaft. Check that the notch on the vibration damper lines up with the arrow on the lower timing chain cover.",
    tasks: [
      { type: "measure", spec: "intakeMark", label: "Head ridge edge to intake camshaft mark", items: ["Reading"] },
      { type: "measure", spec: "camToCam", label: "Exhaust camshaft mark to intake camshaft mark", items: ["Reading"] },
      { type: "check", items: ["Vibration damper notch aligned with the arrow on the lower timing chain cover · 15 · p. 103"] },
    ],
  },
  {
    id: "results",
    group: "close",
    title: "What the readings mean",
    figure: "front",
    focus: ["camChain", "camTensioner"],
    ref: "15 · p. 103",
    say: "One tooth of offset changes the reading by about 6 mm. If the timing is off by a tooth, the manual says the timing chain must be installed again. A reading near the end of the range on an engine with the symptoms points to a worn chain and tensioner: the chain, tensioner, rails and sprockets are then usually replaced together.",
    tasks: [
      {
        type: "choice",
        q: "The cam-to-cam reading is 131 mm. What does it mean?",
        options: [
          { text: "Slightly out, but fine to drive", ok: false, why: "131 mm is about 6 mm over: that is a tooth of offset." },
          { text: "About one tooth of offset: the chain must be installed again", ok: true, why: "The manual gives about 6 mm per tooth (15 · p. 103). Find out why it jumped: tensioner and chain wear." },
          { text: "The gauge was set up wrong, so ignore it", ok: false, why: "Re-check the setup if in doubt, but don't ignore a reading this far out." },
        ],
      },
      { type: "note", text: "Replacing the timing chain follows the chain drive procedure in Rep. Gr. 15; the engine assembly job on this site walks through installing the chains, tensioners and timing marks." },
    ],
  },
  {
    id: "reassemble",
    group: "close",
    title: "Reassemble",
    figure: "ignition",
    focus: ["plugs", "coils"],
    install: ["plugs", "coils"],
    ref: "15 · p. 69, 28 · p. 253",
    say: "Take out the dial gauge, refit the spark plug and coil, then the upper timing chain cover with a new seal and O-ring.",
    tasks: [
      { type: "torque", what: "Spark plug, cylinder 1", stages: [{ bolts: "Spark plug", nm: 30 }] },
      { type: "check", items: ["Ignition coil pushed fully on, connector clicked in"] },
      { type: "order", diagram: "upperCover5", q: "Upper timing chain cover: click the five bolts in order." },
      { type: "torque", what: "Upper timing chain cover bolts", stages: [{ bolts: "Bolts 1–5, in sequence", nm: 9 }] },
      { type: "check", items: ["Front noise insulation refitted", "Fault codes cleared, engine started, no new codes"] },
    ],
  },
];

export function build() {
  const figs = buildFigures();
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: SPEC, diagrams: { upperCover5: DIAGRAMS.upperCover5 }, figures: { front: figs.front, ignition: ignition() } };
}
