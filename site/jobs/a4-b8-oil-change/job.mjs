// Audi A4 (B8) 2.0 TFSI engine oil and filter change: a DRAFT job. General workshop procedure;
// the oil standard, capacity and torques are `pending` until licensed data is loaded.
import { sump } from "./figures.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";

export const meta = {
  id: "a4-b8-oil-change",
  title: "Engine oil and filter change",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" },
  engine: "2.0 TFSI (EA888, e.g. CAEB)",
  system: "Service",
  vehicleLine: "Audi A4 (B8, 2008–2016), 2.0 TFSI petrol engine",
  status: "draft",
  private: false,
  minutes: 60,
  difficulty: "Beginner",
  summary: "Drain the old oil, change the filter, fit the drain plug with a new seal, refill to the right level and reset the service reminder.",
  source: { name: "Licensed repair data (not loaded yet)", prefix: "Source: " },
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the oil standard, capacity and torques show as “spec not loaded”. Take them from your licensed repair data for your engine code.",
};

const VARIANTS = {
  filter: {
    q: "Oil filter type",
    help: "Look at the filter on the engine, or at the new one: a spin-on filter is a sealed metal can; a cartridge is a paper element that sits in a plastic or metal housing with a screw-off cap.",
    options: [
      { id: "spin", label: "Spin-on can" },
      { id: "cartridge", label: "Cartridge in a housing" },
    ],
  },
};

const SPEC = {};
const pending = (bolts) => ({ bolts, pending: true });

const GROUPS = [
  { id: "prep", name: "Preparation" },
  { id: "drain", name: "Drain and filter" },
  { id: "fill", name: "Refill and check" },
];

const STEPS = [
  {
    id: "parts",
    group: "prep",
    title: "Get the right oil and parts",
    figure: "sump",
    focus: ["filter"],
    ref: "",
    say: "Modern engines need oil that meets the car maker's standard, not just the right viscosity: the standard number (a VW 50x xx approval for this engine) is printed on the bottle. The amount depends on the engine and whether the filter is changed.",
    tasks: [
      { type: "variant", keys: ["filter"] },
      { type: "check", items: ["Oil meeting the standard your source gives for this engine code (spec not loaded)", "Oil quantity from your source, with the filter changed (spec not loaded)", "New oil filter for this engine", "New drain plug seal, or a new drain plug if your source says the plug is single-use", "Drain pan big enough for the full oil quantity, funnel, rags, gloves"] },
      { type: "check", when: { filter: "cartridge" }, items: ["New O-rings for the filter housing cap (usually in the filter box)"] },
    ],
  },
  {
    id: "warm",
    group: "prep",
    title: "Warm the engine and raise the car",
    figure: "sump",
    focus: ["pan"],
    ref: "",
    say: "Warm oil drains faster and carries more dirt out with it. A few minutes' running is enough: oil at full running temperature can scald you. The car must sit level for the oil to drain fully and for the level check afterwards.",
    tasks: [
      { type: "check", items: ["Engine run for a few minutes, then switched off", "Car raised level on ramps or on axle stands at the jacking points", "Wheels chocked and parking brake on"] },
      {
        type: "choice",
        q: "Why should the car be level when you drain and refill?",
        options: [
          { text: "So the oil drains fully and the dipstick reads true", ok: true, why: "A tilted pan holds back old oil, and the level reads wrong on a slope." },
          { text: "It doesn't matter for an oil change", ok: false, why: "It does: the drain plug is at one point of the pan, and the dipstick reading depends on the car being level." },
          { text: "To protect the turbocharger", ok: false, why: "The turbo isn't affected by how the car stands while it is off." },
        ],
      },
    ],
  },
  {
    id: "undertray",
    group: "prep",
    title: "Open the filler cap and remove the undertray",
    figure: "bay",
    focus: ["oilCap"],
    remove: ["oilCap"],
    ref: "",
    say: "Take the oil filler cap off: air coming in from the top lets the oil drain freely. Under the car, take off the engine undertray (the noise insulation panel) to reach the drain plug.",
    tasks: [{ type: "check", items: ["Oil filler cap off, laid somewhere clean", "Undertray off, its screws kept together"] }],
  },
  {
    id: "drain",
    group: "drain",
    title: "Drain the old oil",
    figure: "sump",
    focus: ["drainPlug", "sealRing"],
    remove: ["drainPlug", "sealRing"],
    ref: "",
    say: "Put the drain pan under the plug, a little to the side the oil will shoot towards. Loosen the plug with a ring spanner, then unscrew the last turns by hand, pushing it in, and pull it away quickly. Let the oil drain until it only drips.",
    tasks: [
      { type: "check", items: ["Drain pan in place", "Plug out, oil draining", "Old seal ring off the plug (it may stick to the pan)", "Plug threads and pan sealing face wiped clean and checked for damage"] },
      {
        type: "choice",
        q: "The old oil looks milky, like coffee with milk. What does it mean?",
        options: [
          { text: "Normal for an engine that only makes short trips", ok: false, why: "A little condensation under the filler cap can come from short trips, but milky oil in the pan is not normal." },
          { text: "Coolant may be getting into the oil", ok: true, why: "That points to a fault such as a head gasket or oil cooler. Don't just refill: have it diagnosed before driving." },
          { text: "The oil was the wrong viscosity", ok: false, why: "Viscosity doesn't change the oil's colour like that." },
        ],
      },
    ],
  },
  {
    id: "filter-off",
    group: "drain",
    title: "Take the old filter off",
    figure: "sump",
    focus: ["filter"],
    remove: ["filter"],
    ref: "",
    say: "Put rags or a tray under the filter: oil runs out as it comes off. Unscrew it with a filter wrench or the cap socket.",
    tasks: [
      { type: "check", when: { filter: "spin" }, items: ["Filter unscrewed and drained into the pan", "Old filter's rubber seal came off with it, not left stuck on the engine", "Mounting face wiped clean"] },
      { type: "check", when: { filter: "cartridge" }, items: ["Housing cap unscrewed and old element pulled out", "Old O-rings removed from the cap", "Inside of the cap wiped clean"] },
    ],
  },
  {
    id: "filter-on",
    group: "drain",
    title: "Fit the new filter",
    figure: "sump",
    focus: ["filter"],
    install: ["filter"],
    ref: "",
    say: "Wipe a thin film of new oil on the new seal so it seats without twisting. Tighten to the value in your source: too tight crushes the seal and makes the filter hard to remove next time.",
    tasks: [
      { type: "check", when: { filter: "spin" }, items: ["New filter's seal oiled", "Filter screwed on by hand until the seal touches"] },
      { type: "check", when: { filter: "cartridge" }, items: ["New O-rings fitted to the cap and lightly oiled", "New element pushed into the cap or housing as your source shows"] },
      { type: "torque", what: "Oil filter or filter housing cap", stages: [pending("Filter or cap")] },
    ],
  },
  {
    id: "plug-in",
    group: "drain",
    title: "Fit the drain plug with a new seal",
    figure: "sump",
    focus: ["drainPlug", "sealRing"],
    install: ["sealRing", "drainPlug"],
    ref: "",
    say: "A crushed seal doesn't seal twice. Fit the new seal ring, screw the plug in by hand to be sure it isn't cross-threaded, then torque it. The pan is aluminium: an overtightened plug strips its thread.",
    tasks: [
      { type: "check", items: ["New seal ring on the plug", "Plug started by hand, at least two turns, without force"] },
      { type: "torque", what: "Oil drain plug", tool: "torque wrench", stages: [pending("Drain plug")] },
    ],
  },
  {
    id: "fill",
    group: "fill",
    title: "Fill with new oil",
    figure: "bay",
    focus: ["oilCap"],
    install: ["oilCap"],
    ref: "",
    say: "Pour in a little less than the full quantity, then fit the filler cap. Run the engine for a minute, so the filter fills, and check for leaks at the plug and filter. Switch off and wait a few minutes for the oil to settle back into the pan.",
    tasks: [
      { type: "check", items: ["Oil poured in, about half a litre short of the full quantity", "Filler cap on", "Engine run for about a minute; oil warning light went out within seconds", "No leaks at the drain plug or filter", "Engine off, waited a few minutes"] },
      {
        type: "choice",
        q: "Why not put the full quantity in at once?",
        options: [
          { text: "Some old oil always stays in the engine, so the full quantity can overfill it", ok: true, why: "Fill short, check, then top up. Too much oil can foam and damage the catalytic converter or seals." },
          { text: "New oil expands when warm", ok: false, why: "It does a little, but that's not the reason: residual old oil is." },
          { text: "To save oil", ok: false, why: "The engine needs the right level, not less." },
        ],
      },
    ],
  },
  {
    id: "level",
    group: "fill",
    title: "Check the level and top up",
    figure: "bay",
    focus: ["dipstick"],
    ref: "",
    say: "Pull the dipstick, wipe it, push it fully back in and pull it again to read. Top up in small amounts until the level is in the upper part of the marked range, without going over the top mark. Cars without a dipstick show the level in the dashboard menu.",
    tasks: [{ type: "check", items: ["Level read on the dipstick (or in the car's menu)", "Topped up in small amounts, level in the marked range, not above it", "Dipstick and filler cap secure"] }],
  },
  {
    id: "finish",
    group: "fill",
    title: "Finish: undertray, service reminder, old oil",
    figure: "sump",
    focus: ["pan"],
    ref: "",
    say: "Refit the undertray and lower the car. Reset the service reminder as your source describes for your car (through the instrument cluster or the MMI menu, or with a diagnostic tool), so the car counts down to the next change.",
    tasks: [{ type: "check", items: ["Undertray refitted, all screws in", "Car lowered", "Service reminder reset", "Date and mileage noted in the service record", "Old oil and filter taken to a collection point: never into a drain or household rubbish"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: SPEC, diagrams: {}, figures: { sump: sump(), bay: bay() } };
}
