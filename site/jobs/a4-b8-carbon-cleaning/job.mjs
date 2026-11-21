// Audi A4 (B8) 2.0 TFSI intake valve carbon cleaning (walnut blasting): a DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { buildFigures } from "../../../docs/figures/ea888-caeb/rebuild/figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-carbon-cleaning",
  title: "Intake valve carbon cleaning",
  engine: "2.0 TFSI (EA888, e.g. CAEB)",
  vehicleLine: "Audi A4 (B8, 2008–2016), 2.0 TFSI gasoline engine",
  system: "Engine",
  commonProblem: "Carbon build-up on the intake valves",
  minutes: 360,
  difficulty: "Intermediate–expert",
  summary: "Direct injection means fuel never washes the intake valves, so carbon builds up. Remove the intake manifold, close each cylinder's valves, and blast the carbon off with crushed walnut shell.",
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the intake manifold removal, gaskets and torques show as “spec not loaded”. Your EA888 repair manual covers the intake manifold in its own repair group if you have it.",
});

const GROUPS = [{ id: "why", name: "Symptoms" }, { id: "open", name: "Open up" }, { id: "clean", name: "Clean each cylinder" }, { id: "close", name: "Close up" }];
const CYL = ["Cylinder 1", "Cylinder 2", "Cylinder 3", "Cylinder 4"];

const STEPS = [
  {
    id: "why",
    group: "why",
    title: "Why carbon builds up",
    figure: "valve",
    focus: ["valve"],
    ref: "",
    say: "On a direct injection engine the fuel goes straight into the cylinder, so nothing washes the back of the intake valves. Oil mist from the crankcase breather bakes onto them. Commonly reported symptoms: rough idle and misfires when cold, hesitation, loss of power and worse fuel economy, often from around 50,000 to 80,000 miles.",
    tasks: [
      { type: "check", items: ["Misfire codes and when they happen noted", "Spark plugs and coils ruled out first (see that job)", "Crankcase breather checked (a failed one makes carbon build-up worse)", "Valves seen with a borescope through the intake port if possible"] },
      {
        type: "choice",
        q: "Will a fuel additive clean the intake valves on this engine?",
        options: [
          { text: "Yes, with every tank", ok: false, why: "The fuel doesn't touch the intake valves on a direct injection engine, so additives in the tank can't reach them." },
          { text: "No: the carbon has to be removed mechanically or with a chemical through the intake", ok: true, why: "Walnut blasting with the manifold off is the most thorough method." },
          { text: "Only premium fuel works", ok: false, why: "The fuel still doesn't reach the valves." },
        ],
      },
    ],
  },
  {
    id: "manifold",
    group: "open",
    title: "Remove the intake manifold",
    figure: "head",
    focus: ["headBody"],
    ref: "",
    say: "Disconnect the battery, then remove the intake manifold as your source describes, with its sensors, wiring and vacuum lines labeled. Cover the open fuel injectors and anything else you can't seal off: walnut grit must not get anywhere except the port you're cleaning.",
    tasks: [{ type: "check", items: ["Battery disconnected (negative first)", "Intake manifold removal procedure from your source (spec not loaded)", "Every connector and hose labeled or photographed", "Manifold off; ports covered with tape until each is cleaned", "New manifold gaskets ready (spec not loaded)"] }],
  },
  {
    id: "valves-closed",
    group: "clean",
    title: "Close the valves of the cylinder you're cleaning",
    figure: "valve",
    focus: ["valve", "spring"],
    ref: "",
    say: "Turn the crankshaft by hand, in the direction of engine rotation, until both intake valves of the cylinder you're cleaning are closed. That cylinder's piston is then near top dead center on compression. Otherwise grit falls into the cylinder.",
    tasks: [
      { type: "check", items: ["Crank turned only in the direction of rotation", "Both intake valves of this cylinder closed (seen through the port)"] },
      {
        type: "choice",
        q: "Why must the intake valves be closed before blasting?",
        options: [
          { text: "So the grit can't fall into the cylinder", ok: true, why: "With the valves shut, only the port and valve backs are exposed." },
          { text: "To make the carbon softer", ok: false, why: "Valve position doesn't change the carbon." },
          { text: "It doesn't matter", ok: false, why: "Grit in the cylinder scores the bore and has to be removed." },
        ],
      },
    ],
  },
  {
    id: "blast",
    group: "clean",
    title: "Blast each cylinder's ports",
    figure: "valve",
    focus: ["valve"],
    ref: "",
    say: "With the blaster's nozzle sealed in the port and the vacuum running, blast in short bursts until the valve backs and port are clean metal. Vacuum out all the grit, then check with a light. Repeat for each cylinder, turning the crank to close its valves each time.",
    tasks: [
      { type: "check", items: CYL.map((c) => `${c}: valves closed, ports blasted clean, all grit vacuumed out`) },
      { type: "check", items: ["Ports wiped clean and checked with a light and borescope", "No grit left anywhere around the ports or on the injectors"] },
    ],
  },
  {
    id: "close",
    group: "close",
    title: "Refit the manifold and test",
    figure: "head",
    focus: ["headBody"],
    ref: "",
    say: "Fit new manifold gaskets and the manifold, torque its bolts, and reconnect everything you labeled. Reconnect the battery last. Clear fault codes and test drive.",
    tasks: [
      { type: "torque", what: "Intake manifold bolts", stages: [pending("Manifold bolts, in your source's sequence")] },
      { type: "check", items: ["All connectors and vacuum lines reconnected", "Battery reconnected (negative last)", "Engine starts and idles smoothly; fault codes cleared", "Test drive: no misfires, smoother idle"] },
    ],
  },
];

export function build() {
  const figs = buildFigures();
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { valve: figs.valve, head: figs.head } };
}
