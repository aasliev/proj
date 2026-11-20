// Audi A4 (B8) quick fluid check and top-up: engine oil, coolant, brake fluid, washer fluid.
// A DRAFT job: the fluid specifications come from licensed data, which is not loaded yet.
import { bay } from "./figures.mjs";

export const meta = {
  id: "a4-b8-fluid-check",
  title: "Check and top up fluids",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" },
  system: "Service",
  vehicleLine: "Audi A4 (B8, 2008–2016)",
  status: "draft",
  private: false,
  minutes: 15,
  difficulty: "Beginner",
  summary: "Check the engine oil, coolant, brake fluid and washer fluid, top up the right fluid where needed, and know when a low level means a fault.",
  source: { name: "Licensed repair data (not loaded yet)", prefix: "Source: " },
  draftNote: "The checks are general workshop practice. No licensed data is loaded yet, so the exact oil, coolant and brake fluid specifications show as “spec not loaded”: take them from your licensed data or the owner's manual.",
};

const GROUPS = [
  { id: "prep", name: "Before you start" },
  { id: "check", name: "Checks" },
];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Park level, engine off",
    figure: "bay",
    focus: ["engine"],
    ref: "",
    say: "Park on level ground with the engine off. Find each filler by the symbol on its cap: an oil can for engine oil, a thermometer in liquid for coolant, a circle in brackets for brake fluid, a windscreen with a spray for washer fluid.",
    tasks: [
      { type: "check", items: ["Car on level ground, engine off, parking brake on", "Each cap found by its symbol"] },
      {
        type: "choice",
        q: "The engine has just been driven hard. Which check must wait until it has cooled?",
        options: [
          { text: "Washer fluid", ok: false, why: "The washer tank isn't under pressure or hot: you can top it up any time." },
          { text: "Coolant", ok: true, why: "The cooling system is under pressure when hot. Opening the cap can spray scalding coolant. Wait until the engine is cold." },
          { text: "Brake fluid", ok: false, why: "The brake reservoir is not pressurised; check it any time, though it is best done on level ground." },
        ],
      },
    ],
  },
  {
    id: "oil",
    group: "check",
    title: "Engine oil level",
    figure: "bay",
    focus: ["dipstick", "oilCap"],
    ref: "",
    say: "With the engine warm, switch off and wait a few minutes so the oil drains back. Pull the dipstick, wipe it, push it fully home and pull it again. Some cars have no dipstick and show the level in the dashboard menu instead.",
    tasks: [
      { type: "check", items: ["Level read: in the marked range", "If low: topped up in small amounts with oil meeting the standard for your engine (spec not loaded), rechecked after each top-up", "Filler cap and dipstick secure"] },
      {
        type: "choice",
        q: "You top up a litre between every service. What should you do?",
        options: [
          { text: "Nothing: all engines use that much", ok: false, why: "Some oil use is normal, but if you're adding a lot regularly, check what your source allows, and have it looked at if it's more." },
          { text: "Note how much and how often, and compare with the consumption your source allows", ok: true, why: "Keep a record. High oil use can point to worn piston rings, valve stem seals or the crankcase breather." },
          { text: "Switch to a thicker oil", ok: false, why: "Use only oil that meets the engine's standard; thicker oil hides the problem and can harm the engine." },
        ],
      },
    ],
  },
  {
    id: "coolant",
    group: "check",
    title: "Coolant level",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Check with the engine cold. The level in the translucent expansion tank should be between its MIN and MAX marks. Top up only with the coolant type your source specifies, mixed as it says: different coolant types can react with each other and form sludge.",
    tasks: [
      { type: "check", items: ["Engine cold", "Level between MIN and MAX", "If low: topped up with the specified coolant (spec not loaded), cap closed tightly", "A level that drops again means a leak: look for wet patches or white crusts on hoses and under the car"] },
    ],
  },
  {
    id: "brake",
    group: "check",
    title: "Brake fluid level",
    figure: "bay",
    focus: ["brakeReservoir"],
    ref: "",
    say: "The level should be between MIN and MAX. As pads wear, the pistons move out and the level drops a little: that is normal. A sudden drop, or a level below MIN, can mean a leak; have the brakes checked before driving.",
    tasks: [
      { type: "check", items: ["Level between MIN and MAX", "Fluid colour noted: fresh fluid is light amber, old fluid is dark", "If topping up: only fluid of the type printed on the cap or given in your source (spec not loaded), from a sealed bottle"] },
      {
        type: "choice",
        q: "The brake fluid is a little below MAX and the pads are half worn. What do you do?",
        options: [
          { text: "Top up to MAX", ok: false, why: "When new pads go in, the pistons are pushed back and the extra fluid can overflow the reservoir." },
          { text: "Leave it: the drop matches the pad wear", ok: true, why: "A slow drop with pad wear is expected. Check for leaks only if it falls fast or below MIN." },
          { text: "Replace the fluid at once", ok: false, why: "The fluid is changed by time interval, not because the level has dropped with pad wear." },
        ],
      },
    ],
  },
  {
    id: "washer",
    group: "check",
    title: "Washer fluid",
    figure: "bay",
    focus: ["washerCap"],
    remove: ["washerCap"],
    ref: "",
    say: "Fill with screenwash mixed as the bottle says. In winter use a mix rated for the coldest temperature you expect: plain water freezes and can crack the tank and pump. Never use washing-up liquid: it smears and can damage the paint and the wiper rubbers.",
    tasks: [{ type: "check", items: ["Tank filled with screenwash at the right strength for the season", "Cap back on", "Washers tested: both jets hit the screen (and the headlamp washers work, if fitted)"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { bay: bay() } };
}
