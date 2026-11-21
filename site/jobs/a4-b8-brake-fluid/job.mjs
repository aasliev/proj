// Audi A4 (B8) brake fluid change: a DRAFT job. General workshop procedure; the fluid type,
// quantity, bleed order and bleed screw torque come from licensed data, which is not loaded yet.
import { buildFigures } from "../a4-b8-front-brakes/figures.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";
import { wheelStar } from "../../lib/diagrams.mjs";

export const meta = {
  id: "a4-b8-brake-fluid",
  title: "Brake fluid change",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" },
  system: "Brakes",
  vehicleLine: "Audi A4 (B8, 2008–2016)",
  status: "draft",
  private: false,
  minutes: 75,
  difficulty: "Intermediate",
  summary: "Flush old brake fluid out through each caliper's bleed screw until clean fluid comes out, without letting air into the system.",
  source: { name: "Licensed repair data (not loaded yet)", prefix: "Source: " },
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the fluid specification, the order to bleed the wheels in and the torques show as “spec not loaded”. Take them from your licensed data: the bleed order is not guessed here.",
};

const VARIANTS = {
  method: {
    q: "How will you push the fluid through?",
    help: "A pressure bleeder on the reservoir pushes fluid through while you work alone. The pedal method needs a helper to press the brake pedal.",
    options: [
      { id: "pressure", label: "Pressure bleeder" },
      { id: "pedal", label: "Helper on the pedal" },
    ],
  },
  transmission: {
    q: "Transmission",
    help: "On manual cars the clutch shares the brake fluid reservoir, so the clutch is bled too.",
    options: [
      { id: "manual", label: "Manual (clutch pedal)" },
      { id: "auto", label: "Automatic" },
    ],
  },
};

const GROUPS = [
  { id: "prep", name: "Preparation" },
  { id: "bleed", name: "Bleeding" },
  { id: "finish", name: "Finish" },
];

const pending = (bolts) => ({ bolts, pending: true });
const WHEELS = ["first", "second", "third", "fourth"];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Fluid, tools and why it is changed",
    figure: "bay",
    focus: ["brakeReservoir"],
    ref: "",
    say: "Brake fluid absorbs water from the air over time. Water lowers its boiling point, so under hard braking it can boil and the pedal goes soft. That's why it is changed by time, not mileage.",
    tasks: [
      { type: "variant", keys: ["method", "transmission"] },
      { type: "check", items: ["New fluid of the type your source gives (spec not loaded), in sealed bottles, enough to flush the whole system (spec not loaded)", "Box-end wrench for the bleed screws, clear tube that fits them snugly, catch bottle", "Syringe or clean baster for the reservoir", "Rags and water to wash off any spills: brake fluid strips paint", "Eye protection and gloves"] },
      {
        type: "choice",
        q: "An opened bottle of brake fluid from last year is in the garage. Can you use it?",
        options: [
          { text: "Yes, if the cap was on", ok: false, why: "Once opened it has already started absorbing moisture." },
          { text: "No: use fluid from a sealed bottle", ok: true, why: "Fresh, sealed fluid is the whole point of the job." },
          { text: "Yes, mixed with new fluid", ok: false, why: "Mixing still adds moisture-laden fluid." },
        ],
      },
    ],
  },
  {
    id: "access",
    group: "prep",
    title: "Raise the car and reach the bleed screws",
    figure: "corner",
    focus: ["wheel"],
    remove: ["bolts", "wheel"],
    ref: "",
    say: "Raise the car level onto axle stands. Take the wheels off if you need room to reach the bleed screws on the calipers. Clean around each bleed screw and check it turns before you start: a seized screw is better found now than with the reservoir half empty.",
    tasks: [{ type: "check", items: ["Car level on axle stands, wheels chocked", "Wheels off where needed", "Each bleed screw cleaned, its dust cap off, and checked that it turns"] }],
  },
  {
    id: "reservoir",
    group: "prep",
    title: "Empty and refill the reservoir",
    figure: "bay",
    focus: ["brakeReservoir"],
    ref: "",
    say: "Draw the old fluid out of the reservoir with a syringe, down to just above the bottom, never empty. Refill with new fluid to MAX. Now the old fluid only has to be pushed out of the pipes.",
    tasks: [{ type: "check", items: ["Old fluid drawn off, not below the reservoir's outlets", "Refilled with new fluid to MAX", "Pressure bleeder fitted and set to the pressure its maker gives"], }],
  },
  {
    id: "order",
    group: "bleed",
    title: "Find the bleed order",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    say: "Cars with ABS and stability control have a set order to bleed the wheels in. Many manuals start with the wheel farthest from the master cylinder, but not all: use the order your source gives for this car.",
    tasks: [
      { type: "check", items: ["Bleed order written down from your source (spec not loaded)"] },
      {
        type: "choice",
        q: "What happens if the reservoir runs dry during bleeding?",
        options: [
          { text: "Nothing, just top it up", ok: false, why: "Air is drawn in and can get into the ABS unit." },
          { text: "Air gets into the system, possibly into the ABS unit", ok: true, why: "Air in the ABS unit may need a diagnostic tool to purge it. Keep checking the level and never let it fall below MIN." },
          { text: "The ABS warning light simply resets", ok: false, why: "The problem is air in the system, not a warning light." },
        ],
      },
    ],
  },
  ...WHEELS.map((which, i) => ({
    id: `bleed-${i + 1}`,
    group: "bleed",
    title: `Bleed the ${which} wheel in your order`,
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    say: i === 0 ? "Push the tube onto the bleed screw with its end in the bottle, below the fluid already in it so air can't be drawn back. Open the screw a quarter turn and let fluid run until it comes out clean and bubble-free, then close it." : "Same as before: tube on, screw open, run until the fluid is clean and bubble-free, close. Check the reservoir level before every wheel.",
    tasks: [
      { type: "check", when: { method: "pressure" }, items: ["Reservoir level checked, topped up if needed", "Tube on the bleed screw, end under the fluid in the bottle", "Screw opened; fluid ran clean with no bubbles", "Screw closed before taking the tube off"] },
      { type: "check", when: { method: "pedal" }, items: ["Reservoir level checked, topped up if needed", "Tube on the bleed screw, end under the fluid in the bottle", "Helper pressed the pedal and held it; screw opened, then closed before the pedal was released; repeated until the fluid ran clean with no bubbles", "Helper presses smoothly, not all the way to the floor if your source warns against it"] },
      { type: "torque", what: "Bleed screw", stages: [pending("Bleed screw")] },
    ],
  })),
  {
    id: "clutch",
    group: "bleed",
    title: "Bleed the clutch",
    figure: "bay",
    focus: ["brakeReservoir"],
    ref: "",
    when: { transmission: "manual" },
    say: "The clutch slave cylinder takes its fluid from the same reservoir. Bleed it at its bleed screw the way your source describes, until the fluid is clean, and pump the clutch pedal afterwards to check it returns properly.",
    tasks: [{ type: "check", items: ["Clutch slave cylinder bled until the fluid ran clean", "Clutch pedal returns fully and feels normal"] }],
  },
  {
    id: "finish",
    group: "finish",
    title: "Level, pedal, wheels and test",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    say: "Take the pressure bleeder off, set the reservoir level to MAX and fit the cap. The pedal must be firm and not sink under steady pressure before the car moves. Then refit the wheels and test the brakes at low speed first.",
    tasks: [
      { type: "check", items: ["Reservoir at MAX, cap on", "Bleed screw dust caps on, no leaks at any screw", "Pedal firm and doesn't sink under steady pressure", "Spilt fluid washed off with water", "Wheels on, bolts in by hand, car lowered"] },
      { type: "order", diagram: "wheel", q: "Tighten each wheel's bolts in a star pattern, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "torque wrench", stages: [pending("All five, in the star pattern")] },
      { type: "check", items: ["Brakes tested at walking pace, then gently on the road", "Old fluid taken to a collection point", "Date noted for the next change"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: { wheel: wheelStar() }, figures: { corner: buildFigures().corner, bay: bay() } };
}
