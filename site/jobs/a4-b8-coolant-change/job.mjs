// Audi A4 (B8) coolant drain, refill and bleed: a DRAFT job.
import { draftMeta } from "../../lib/a4b8.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-coolant-change",
  title: "Coolant drain, refill and bleed",
  system: "Cooling",
  minutes: 120,
  difficulty: "Intermediate",
  summary: "Drain the old coolant, refill with the right coolant mixed with distilled water, and get all the air out so the engine doesn't overheat.",
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the coolant type, quantity, drain points and bleed procedure show as “spec not loaded”.",
});

const VARIANTS = {
  filler: {
    q: "How will you fill?",
    help: "A vacuum filler pulls a vacuum on the system, then lets coolant in, so almost no air is trapped. Filling by hand works but needs more bleeding.",
    options: [
      { id: "vacuum", label: "Vacuum filler" },
      { id: "hand", label: "By hand" },
    ],
  },
};

const GROUPS = [{ id: "prep", name: "Preparation" }, { id: "drain", name: "Drain" }, { id: "fill", name: "Fill and bleed" }];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Cold engine, the right coolant",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Only work on a cold engine: hot coolant is under pressure and can scald. Use the coolant type your source gives, mixed with distilled water. Mixing coolant types can form sludge.",
    tasks: [
      { type: "variant", keys: ["filler"] },
      { type: "check", items: ["Engine cold", "Coolant to your source's specification and enough for the system (spec not loaded)", "Distilled water for mixing", "Refractometer or tester for freeze protection", "Drain pan, and new clips or O-rings for any hose you open"] },
    ],
  },
  {
    id: "drain",
    group: "drain",
    title: "Drain the coolant",
    figure: "bay",
    focus: ["coolantTank"],
    remove: [],
    ref: "",
    say: "Open the expansion tank cap slowly. Under the car, open the drain point your source gives (a drain plug or a lower radiator hose) and let it drain into the pan. Coolant is poisonous and tastes sweet to animals: clean up spills at once.",
    tasks: [
      { type: "check", items: ["Expansion tank cap opened slowly", "Splash shield off", "Drain point from your source opened (spec not loaded)", "All coolant drained into the pan", "Spills cleaned up; pan out of reach of children and pets"] },
      {
        type: "choice",
        q: "Can the old coolant be poured down a drain?",
        options: [
          { text: "Yes, it's mostly water", ok: false, why: "Coolant is toxic. Take it to a collection point or a shop that accepts it." },
          { text: "No: take it to a collection point", ok: true, why: "Many parts stores and recycling centers accept used coolant." },
        ],
      },
    ],
  },
  {
    id: "close-drain",
    group: "drain",
    title: "Close the drain",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Close the drain point or refit the hose with its clip, using new O-rings where your source says.",
    tasks: [{ type: "check", items: ["Drain closed, hose clip fully locked", "New O-ring fitted where your source requires one"] }],
  },
  {
    id: "fill",
    group: "fill",
    title: "Fill",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Mix the coolant with distilled water to the freeze protection your source gives. Then fill.",
    tasks: [
      { type: "check", when: { filler: "vacuum" }, items: ["Vacuum filler fitted to the expansion tank", "Vacuum pulled and held for the time the tool says (no drop = no leaks)", "Coolant drawn in until the vacuum is gone"] },
      { type: "check", when: { filler: "hand" }, items: ["Filled slowly through the expansion tank to the MAX mark", "Any bleed screws your source lists opened while filling and closed when coolant comes out without bubbles"] },
      { type: "check", items: ["Freeze protection checked with a refractometer"] },
    ],
  },
  {
    id: "bleed",
    group: "fill",
    title: "Bleed and check",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Set the heater to maximum. Run the engine until the radiator fan comes on, watching the temperature gauge and the level, as your source's bleed procedure describes. Some cars have an electric coolant pump that a diagnostic tool runs for bleeding. Let it cool fully, then top up to the mark.",
    tasks: [
      { type: "check", items: ["Heater on full", "Engine run to fan cut-in; temperature gauge normal", "Bleed procedure from your source followed (spec not loaded)", "No leaks at the drain or any hose", "Level topped up when cold, the next day too", "Heater blows hot (if not, air is still trapped)"] },
      {
        type: "choice",
        q: "After filling, the heater blows cold and the gauge climbs. What's wrong?",
        options: [
          { text: "The thermostat is faulty", ok: false, why: "Possible, but right after a refill the usual cause is air." },
          { text: "Air is trapped in the system: stop and bleed it", ok: true, why: "An air pocket stops coolant flow through the heater and engine. Switch off before it overheats, let it cool, and bleed again." },
          { text: "The coolant mix is too strong", ok: false, why: "That doesn't cause this right after a refill." },
        ],
      },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { bay: bay() } };
}
