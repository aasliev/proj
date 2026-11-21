// Audi A4 (B8) wheel swap (e.g. summer to winter wheels) with a tire check: a DRAFT job.
// Wheel bolt torque and tire pressures come from licensed data or the car's own label.
// Fitting tires onto rims needs a tire machine and balancer, so it is not part of this job.
import { buildFigures } from "../a4-b8-front-brakes/figures.mjs";
import { wheelStar } from "../../lib/diagrams.mjs";

export const meta = {
  id: "a4-b8-wheel-swap",
  title: "Wheel swap and tire check",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" },
  system: "Wheels and tires",
  vehicleLine: "Audi A4 (B8, 2008–2016)",
  status: "draft",
  private: false,
  minutes: 60,
  difficulty: "Beginner",
  summary: "Swap a set of wheels, for example summer to winter: check the tires, fit each wheel to its position, torque the bolts in a star pattern, set pressures and the tire pressure monitor.",
  source: { name: "Licensed repair data (not loaded yet)", prefix: "Source: " },
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so the wheel bolt torque shows as “spec not loaded”. Tire pressures are on the label inside the fuel door. Putting tires onto rims needs a tire machine and balancing: that is a job for a tire shop.",
};

const VARIANTS = {
  tpms: {
    q: "Tire pressure monitoring",
    help: "Check your owner's manual or the car menu. Some cars work out pressure from wheel speeds and need the new pressures stored after a swap; others have a sensor in each wheel.",
    options: [
      { id: "store", label: "Store pressures in the car menu" },
      { id: "sensors", label: "Sensors in the wheels" },
      { id: "none", label: "None" },
    ],
  },
};

const GROUPS = [
  { id: "prep", name: "Check the wheels" },
  { id: "swap", name: "Swap" },
  { id: "finish", name: "Pressures and finish" },
];

// Not graded: the legal tread minimum depends on the country, pressures come from the car's label.
const SPEC = {
  treadDepth: { pending: true, unit: "/32 in", why: "legal minimum is 2/32 inch in most US states (see below)" },
  pressure: { pending: true, unit: "psi", why: "from the label inside the fuel door" },
};
const pending = (bolts) => ({ bolts, pending: true });
const TIRES = ["Front left", "Front right", "Rear left", "Rear right"];
const tread = TIRES.map((t) => `${t}: shallowest groove`);

const STEPS = [
  {
    id: "inspect",
    group: "prep",
    title: "Check the wheels you're fitting",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "Check the wheels going on before you take the others off. Measure tread depth in the main grooves across the tire and note the lowest reading for each. Check the age: the last four digits of the DOT code are the week and year it was made.",
    tasks: [
      { type: "variant", keys: ["tpms"] },
      { type: "measure", spec: "treadDepth", label: "Tread depth in 32nds of an inch (record the lowest reading)", items: tread },
      { type: "note", text: "In most US states the legal minimum is 2/32 inch (about 1.6 mm). For rain and snow, many tire makers recommend replacing at 4/32 inch or more. Quick check: put a penny in the shallowest groove with Lincoln's head down; if you can see all of his head, the tread is at 2/32 inch or less." },
      { type: "check", items: ["No cuts, bulges or exposed cords on the sidewalls", "Tire age checked from the DOT code", "Tire size and load and speed ratings are allowed for this car (see the tire label or owner's manual)", "Rims not cracked or bent"] },
    ],
  },
  {
    id: "positions",
    group: "prep",
    title: "Plan where each wheel goes",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "Wheels are usually marked with their last position (FL, FR, RL, RR) when stored. Directional tires have an arrow on the sidewall that must point in the direction the wheel turns when driving forward.",
    tasks: [
      { type: "check", items: ["Each wheel marked with its position", "Directional tires checked: arrows point forward on both sides", "Wheels coming off will be marked with their position before storage"] },
      {
        type: "choice",
        q: "Two tires have more tread than the other two. Where should they go?",
        options: [
          { text: "On the rear axle", ok: true, why: "Tire makers generally recommend the better tires at the rear, so the rear keeps grip and the car is less likely to spin on a wet road." },
          { text: "On the front axle, because it steers", ok: false, why: "That feels logical, but a car that loses grip at the rear is much harder to control." },
          { text: "One front and one rear on the same side", ok: false, why: "Tires on the same axle should match; mixed tread on one axle upsets the handling." },
        ],
      },
    ],
  },
  {
    id: "loosen",
    group: "swap",
    title: "Loosen the bolts, raise the car, take the wheel off",
    figure: "corner",
    focus: ["wheel", "bolts"],
    remove: ["bolts", "wheel"],
    ref: "",
    say: "Slacken each bolt half a turn while the wheel is still on the ground. Raise the car at the jacking point and support it on an axle stand. Then take the bolts out and lift the wheel off. Do one wheel at a time, or support the car at every corner you lift.",
    tasks: [{ type: "check", items: ["Parking brake on, transmission in P (or first gear for a manual), other wheels chocked", "Bolts slackened on the ground (locking bolt adapter used if fitted)", "Car on an axle stand at the jacking point", "Bolts out, wheel off"] }],
  },
  {
    id: "clean",
    group: "swap",
    title: "Clean the hub and check the brakes",
    figure: "corner",
    focus: ["hub", "disc"],
    ref: "",
    say: "With the wheel off it's the easiest time to look at the brakes. Clean rust off the hub's centering spigot and contact face with a wire brush so the wheel sits flat.",
    tasks: [{ type: "check", items: ["Hub contact face and centering spigot cleaned with a wire brush", "Pad thickness looked at through the caliper", "Disc surface looked at: no deep grooves or a large lip at the edge", "Brake hose not cracked or chafed", "Wheel bolts clean and undamaged, threads not stretched"] }],
  },
  {
    id: "fit",
    group: "swap",
    title: "Fit the wheel and tighten in a star pattern",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    say: "Lift the wheel onto the spigot, screw all the bolts in by hand, and snug them up evenly. Lower the car so the tire takes weight, then torque in a star pattern with a torque wrench. Never use an impact gun to tighten: it can overtighten and warp the disc.",
    tasks: [
      { type: "check", items: ["Wheel on the spigot, all bolts screwed in by hand", "Bolts snugged evenly, car lowered"] },
      { type: "order", diagram: "wheel", q: "Tighten the wheel bolts in a star pattern: click them in order, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "torque wrench", stages: [pending("All five, in the star pattern")] },
      { type: "check", items: ["Repeated for all four wheels"] },
    ],
  },
  {
    id: "pressures",
    group: "finish",
    title: "Set the tire pressures",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "Set pressures with the tires cold, before driving. The pressures for your tire size and load are on the label inside the fuel door; full load needs higher pressures.",
    tasks: [
      { type: "measure", spec: "pressure", label: "Cold pressure after setting (psi)", items: TIRES },
      { type: "check", items: ["Pressures set to the label in the fuel door for the load you carry", "Valve caps on"] },
    ],
  },
  {
    id: "tpms",
    group: "finish",
    title: "Set up the tire pressure monitor",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    when: { tpms: ["store", "sensors"] },
    say: "Tell the car about the new wheels, or it may warn falsely or miss a real puncture.",
    tasks: [
      { type: "check", when: { tpms: "store" }, items: ["Pressures stored in the car menu after setting them, as the owner's manual describes"] },
      { type: "check", when: { tpms: "sensors" }, items: ["Sensors matched to the car, by driving or with a tool as your source describes", "No warning light after a short drive"] },
    ],
  },
  {
    id: "recheck",
    group: "finish",
    title: "Re-check after a short drive",
    figure: "corner",
    focus: ["bolts"],
    ref: "",
    say: "Wheels can settle on the hub. Re-check the bolt torque after the first short drive, as many tire makers and shops advise.",
    tasks: [{ type: "check", items: ["Bolt torque re-checked after the first short drive (tire shops commonly say 30–60 miles)", "Removed wheels cleaned, marked with their position and stored lying flat or hung up, away from sunlight"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: SPEC, diagrams: { wheel: wheelStar() }, figures: { corner: buildFigures().corner } };
}
