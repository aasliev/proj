// Audi A4 (B8) transmission fluid service: a DRAFT job. The fluid, quantity, temperature
// window and plug torques depend on the transmission and come from licensed data.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { transmission } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-transmission-fluid",
  title: "Transmission fluid service",
  system: "Transmission",
  minutes: 120,
  difficulty: "Intermediate",
  summary: "Drain and refill the transmission fluid, change the filter where there is one, and set the level the way your transmission needs: automatics are checked hot, at a set fluid temperature.",
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the fluid, quantity, temperature window and torques show as “spec not loaded”. Automatic, CVT and dual-clutch transmissions are filled and leveled differently: follow your source for your transmission code.",
});

const VARIANTS = {
  trans: {
    q: "Which transmission does your car have?",
    help: "The transmission code is on the vehicle data sticker (in the service book or under the trunk floor). On the B8 A4: front-wheel-drive automatics are usually the multitronic CVT; quattro automatics are a tiptronic or, in some markets, the S tronic dual-clutch. A clutch pedal means a manual.",
    options: [
      { id: "cvt", label: "multitronic (CVT)" },
      { id: "auto", label: "tiptronic (automatic)" },
      { id: "dsg", label: "S tronic (dual-clutch)" },
      { id: "manual", label: "Manual" },
    ],
  },
};

const GROUPS = [{ id: "prep", name: "Preparation" }, { id: "drain", name: "Drain" }, { id: "fill", name: "Fill and level" }];
const AUTO = ["cvt", "auto", "dsg"];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Identify the transmission and get the right fluid",
    figure: "transmission",
    focus: ["case"],
    ref: "",
    say: "Each transmission needs its own fluid: the wrong one can ruin it. Buy the fluid your source gives for the transmission code, enough for a drain and refill, plus a new drain plug seal and the filter where your transmission has one.",
    tasks: [
      { type: "variant", keys: ["trans"] },
      { type: "check", items: ["Transmission code read from the vehicle data sticker", "Fluid to the specification your source gives for that code (spec not loaded)", "Quantity for a drain and refill (spec not loaded)", "New seals for the plugs; new filter if your transmission has a serviceable one"] },
      { type: "check", when: { trans: AUTO }, items: ["Diagnostic tool that reads the transmission fluid temperature", "Fill adapter or pump that fits the fill hole"] },
    ],
  },
  {
    id: "level-car",
    group: "prep",
    title: "Car level and the fill plug loosened first",
    figure: "transmission",
    focus: ["fillPlug"],
    ref: "",
    say: "The car must be level, on four axle stands or a lift: the level is set at a hole in the case. Before draining anything, check the fill plug comes out. If it's seized after you've drained the fluid, the car can't be driven.",
    tasks: [
      { type: "check", items: ["Car level, on four stands or a lift", "Splash shield off if it covers the plugs"] },
      {
        type: "choice",
        q: "Which plug do you open first?",
        options: [
          { text: "The drain plug, to get the old fluid out", ok: false, why: "If the fill plug then won't come out, you have an empty transmission and no way to fill it." },
          { text: "The fill plug, to be sure it comes out", ok: true, why: "Always loosen the fill plug first; then drain." },
        ],
      },
    ],
  },
  {
    id: "drain",
    group: "drain",
    title: "Drain the old fluid",
    figure: "transmission",
    focus: ["drainPlug"],
    remove: ["drainPlug"],
    ref: "",
    say: "Drain into a clean pan so you can see what comes out. Some automatic drain plugs have an overflow tube inside them that sets the level: don't lose or damage it.",
    tasks: [
      { type: "check", items: ["Drain plug out, fluid draining into a clean pan", "Old fluid looked at: dark and burnt-smelling, or full of metal particles, means have the transmission checked", "Drain plug cleaned; magnet (if any) wiped"] },
      { type: "check", when: { trans: AUTO }, items: ["Overflow tube in the plug or case noted and kept as your source describes"] },
    ],
  },
  {
    id: "filter",
    group: "drain",
    title: "Change the filter",
    figure: "transmission",
    focus: ["filter"],
    ref: "",
    when: { trans: ["cvt", "dsg"] },
    say: "If your transmission has an external or easy-to-reach filter, change it now with new seals, as your source describes. On some transmissions the filter is inside the oil pan and changed with it.",
    tasks: [{ type: "check", items: ["Filter location and procedure found in your source (spec not loaded)", "Old filter out, new one fitted with new seals"] }, { type: "torque", what: "Filter housing", stages: [pending("Filter housing")] }],
  },
  {
    id: "plug-in",
    group: "fill",
    title: "Fit the drain plug",
    figure: "transmission",
    focus: ["drainPlug"],
    install: ["drainPlug"],
    ref: "",
    say: "Fit the drain plug with a new seal and torque it.",
    tasks: [{ type: "torque", what: "Drain plug (new seal)", stages: [pending("Drain plug")] }],
  },
  {
    id: "fill-manual",
    group: "fill",
    title: "Fill a manual transmission",
    figure: "transmission",
    focus: ["fillPlug"],
    ref: "",
    when: { trans: "manual" },
    say: "Pump fluid into the fill hole until it starts to run out of it, with the car level. That is the correct level. Refit the fill plug with a new seal.",
    tasks: [{ type: "check", items: ["Filled until fluid runs out of the fill hole, car level"] }, { type: "torque", what: "Fill plug (new seal)", stages: [pending("Fill plug")] }],
  },
  {
    id: "fill-auto",
    group: "fill",
    title: "Fill and set the level at temperature",
    figure: "transmission",
    focus: ["fillPlug"],
    ref: "",
    when: { trans: AUTO },
    say: "Pump in fluid, then start the engine and, with the brake held, move through the gears as your source says. Read the fluid temperature on the diagnostic tool. Within the temperature window your source gives, with the engine running, fluid must just run out of the level hole. Then fit the plug.",
    tasks: [
      { type: "check", items: ["Fluid pumped in through the fill or level hole", "Engine started; gears selected as your source describes", "Fluid temperature read on the diagnostic tool, within the window your source gives (spec not loaded)", "With the engine running and at that temperature, fluid just runs out of the level hole"] },
      {
        type: "choice",
        q: "The fluid is already 20 °F above the top of the window when you check the level. What do you do?",
        options: [
          { text: "Set the level anyway", ok: false, why: "Hot fluid expands: setting the level too hot leaves the transmission underfilled when cool." },
          { text: "Let it cool and check again within the window", ok: true, why: "The level is only correct when set within the temperature window your source gives." },
          { text: "Add a little extra to be safe", ok: false, why: "Overfilling is also harmful; set it at the right temperature." },
        ],
      },
      { type: "torque", what: "Fill / level plug (new seal)", stages: [pending("Fill plug")] },
      { type: "check", items: ["Adaptation reset or basic settings done with the diagnostic tool, if your source requires it"] },
    ],
  },
  {
    id: "test",
    group: "fill",
    title: "Test drive and check for leaks",
    figure: "transmission",
    focus: ["case"],
    ref: "",
    say: "Drive gently at first, then check underneath for leaks. Note the date and mileage.",
    tasks: [{ type: "check", items: ["Shifts smoothly, no slipping or warning lights", "No leaks at the plugs or filter", "Splash shield refitted", "Old fluid taken to a collection point", "Date and mileage noted"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { transmission: transmission() } };
}
