// Audi A4 (B8) 2.0 TFSI spark plugs and ignition coils: a DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { ignition } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-spark-plugs",
  title: "Spark plugs and ignition coils",
  engine: "2.0 TFSI (EA888, e.g. CAEB)",
  vehicleLine: "Audi A4 (B8, 2008–2016), 2.0 TFSI petrol engine",
  system: "Service",
  minutes: 60,
  difficulty: "Beginner–intermediate",
  summary: "Pull the four ignition coils, swap the spark plugs one at a time with the engine cold, torque them and refit the coils. Also how to swap coils to find a misfire.",
});

const GROUPS = [{ id: "prep", name: "Preparation" }, { id: "swap", name: "Plugs" }, { id: "finish", name: "Finish" }];
const CYL = ["Cylinder 1", "Cylinder 2", "Cylinder 3", "Cylinder 4"];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Parts, tools and a cold engine",
    figure: "ignition",
    focus: ["coils"],
    ref: "",
    say: "Plugs are threaded into an aluminium head: work only on a cold engine, or the threads can be damaged. Use the plug type your source gives for the engine code: heat range and gap matter on a turbo engine.",
    tasks: [{ type: "check", items: ["Engine cold", "Four plugs of the type your source gives (spec not loaded); gap checked only if your source says to set it", "Spark plug socket with a rubber insert and an extension", "Coil puller, or the special tool your source names", "Torque wrench for the plug torque"] }],
  },
  {
    id: "coils-off",
    group: "prep",
    title: "Take the ignition coils out",
    figure: "ignition",
    focus: ["coils"],
    remove: ["coils"],
    ref: "",
    say: "Remove the engine cover. Unplug each coil's connector by pressing its catch, then pull the coil straight up out of its well. Twisting or levering on the top breaks coils. Keep them in order.",
    tasks: [
      { type: "check", items: ["Engine cover off", "Connectors unplugged, catches pressed, not pulled by the wires", "Coils pulled straight up and laid out in cylinder order"] },
      {
        type: "choice",
        q: "The engine had a misfire on cylinder 2. How can the coils help you find the cause?",
        options: [
          { text: "Swap coil 2 with another cylinder's coil; if the misfire moves, the coil is faulty", ok: true, why: "A misfire that follows the coil points to the coil. If it stays on cylinder 2, look at the plug, injector or compression." },
          { text: "Replace all four coils", ok: false, why: "That may not fix it, and costs more than finding the cause first." },
          { text: "Clean the coil with brake cleaner", ok: false, why: "Cleaning doesn't fix a failing coil." },
        ],
      },
    ],
  },
  {
    id: "plugs-out",
    group: "swap",
    title: "Blow out the wells and remove the plugs",
    figure: "ignition",
    focus: ["plugs"],
    remove: ["plugs"],
    ref: "",
    say: "Blow any dirt out of each plug well with compressed air before you loosen a plug, so nothing falls into the cylinder. Undo each plug with the socket and extension, by hand only at the end. Look at each old plug: its colour tells you how that cylinder is running.",
    tasks: [{ type: "check", items: ["Wells blown clean before loosening", "Plugs out one at a time", "Old plugs compared: even light brown or grey is normal; oily, black and sooty, or white and blistered is not"] }],
  },
  {
    id: "plugs-in",
    group: "swap",
    title: "Fit the new plugs",
    figure: "ignition",
    focus: ["plugs"],
    install: ["plugs"],
    ref: "",
    say: "Start every plug by hand, using the rubber-insert socket on the extension only: it can't force a crossed thread. Then torque it. No anti-seize or grease unless your source says so: it changes the torque.",
    tasks: [
      { type: "check", items: ["Each plug screwed in by hand at least two turns before the wrench"] },
      { type: "torque", what: "Spark plugs", tool: "torque wrench", stages: CYL.map((c) => pending(c)) },
    ],
  },
  {
    id: "coils-on",
    group: "finish",
    title: "Refit the coils and test",
    figure: "ignition",
    focus: ["coils"],
    install: ["coils"],
    ref: "",
    say: "Push each coil down until it seats on the plug, then plug in its connector until the catch clicks. Refit the engine cover. Start the engine: it should idle smoothly with no warning lights.",
    tasks: [{ type: "check", items: ["Coils pushed fully down, each on its own cylinder", "Connectors clicked in", "Engine cover on", "Smooth idle, no engine warning light", "Date and mileage noted"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { ignition: ignition() } };
}
