// Audi A4 (B8) rear brake pads and discs: a DRAFT job. With an electric parking brake the
// caliper pistons are wound back by the parking brake motor, using a diagnostic tool.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { corner } from "../a4-b8-front-brakes/figures.mjs";
import { wheelStar } from "../../lib/diagrams.mjs";

export const meta = draftMeta({
  id: "a4-b8-rear-brakes",
  title: "Rear brake pads and discs",
  system: "Brakes",
  minutes: 150,
  difficulty: "Intermediate",
  summary: "Replace the rear pads and discs, with the electric parking brake put into service mode first so the pistons can be wound back without damage.",
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so torques, wear limits and the parking brake service procedure show as “spec not loaded”. Take them from your licensed data for your brake version.",
});

const VARIANTS = {
  parking: {
    q: "Parking brake",
    help: "A button marked (P) on the center console is an electric parking brake, with a motor on each rear caliper. A lever between the seats is a cable parking brake.",
    options: [
      { id: "electric", label: "Electric (button)" },
      { id: "lever", label: "Lever (cable)" },
    ],
  },
};

const GROUPS = [
  { id: "prep", name: "Preparation" },
  { id: "remove", name: "Removal" },
  { id: "install", name: "Fitting" },
  { id: "finish", name: "Finish" },
];

const STEPS = [
  {
    id: "prep",
    group: "prep",
    title: "Parts, parking brake type and safety",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    say: "The rear brakes hold the parking brake, so how you push the pistons back depends on which kind you have. Find out first, and get the parts for your brake version.",
    tasks: [
      { type: "variant", keys: ["parking"] },
      { type: "check", items: ["Rear pads and discs for your brake version", "New disc retaining screw and any single-use bolts your source lists (spec not loaded)", "Car level on axle stands, front wheels chocked, transmission in P (or first gear for a manual)"] },
      { type: "check", when: { parking: "electric" }, items: ["Diagnostic tool that can put the electric parking brake into service (pad change) mode"] },
      { type: "check", when: { parking: "lever" }, items: ["Piston wind-back tool that fits the caliper piston", "Parking brake lever released"] },
    ],
  },
  {
    id: "service-mode",
    group: "prep",
    title: "Put the electric parking brake into service mode",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    when: { parking: "electric" },
    say: "With the ignition on, use the diagnostic tool to open the parking brake into its service position: the motors wind the pistons fully back. Only then switch the ignition off. Don't press the (P) button until the job is done.",
    tasks: [
      { type: "check", items: ["Service mode started with the diagnostic tool, as your source describes (spec not loaded)", "Motors heard winding back; tool confirms service position", "Ignition off; (P) button left alone"] },
      {
        type: "choice",
        q: "Why not just push the piston back with a clamp, as on the front brakes?",
        options: [
          { text: "It's fine if you push slowly", ok: false, why: "The piston is held by the motor's spindle. Forcing it damages the spindle or motor." },
          { text: "The motor's spindle holds the piston; forcing it breaks the mechanism", ok: true, why: "The motor has to wind it back, using the diagnostic tool's service mode." },
          { text: "The rear pistons are fixed and never move back", ok: false, why: "They move back, but only when the motor winds them." },
        ],
      },
    ],
  },
  {
    id: "wheel-off",
    group: "remove",
    title: "Take the wheel off",
    figure: "corner",
    focus: ["wheel", "bolts"],
    remove: ["bolts", "wheel"],
    ref: "",
    say: "Slacken the bolts on the ground, raise the car onto stands, then take the bolts out and the wheel off.",
    tasks: [{ type: "check", items: ["Bolts slackened on the ground", "Car on axle stands", "Wheel off"] }],
  },
  {
    id: "caliper-off",
    group: "remove",
    title: "Take the caliper off and hang it up",
    figure: "corner",
    focus: ["caliper"],
    remove: ["caliper"],
    ref: "",
    say: "Undo the caliper bolts and lift the caliper off the pads. Hang it from the suspension: its weight must not hang on the brake hose, or on the parking brake wiring or cable.",
    tasks: [{ type: "check", items: ["Caliper bolts out (new ones ready if your source says so)", "Caliper hung up; hose and parking brake wiring or cable not stretched or twisted"] }],
  },
  {
    id: "pads-disc-off",
    group: "remove",
    title: "Pads, carrier and disc off",
    figure: "corner",
    focus: ["innerPad", "outerPad", "carrier", "disc"],
    remove: ["innerPad", "outerPad", "carrier", "disc"],
    ref: "",
    say: "Take out the pads, noting the wear on each. Unbolt the carrier, remove the disc retaining screw and take the disc off. Clean the hub face to bare metal.",
    tasks: [{ type: "check", items: ["Pads out, wear compared", "Carrier off", "Disc off", "Hub face cleaned of rust, carrier pad seats cleaned"] }],
  },
  {
    id: "disc-on",
    group: "install",
    title: "New disc and carrier",
    figure: "corner",
    focus: ["disc", "carrier"],
    install: ["disc", "carrier"],
    ref: "",
    say: "Fit the new disc flat on the clean hub with a new retaining screw, then the carrier.",
    tasks: [{ type: "torque", what: "Disc retaining screw", stages: [pending("Retaining screw")] }, { type: "torque", what: "Carrier bolts", stages: [pending("Carrier bolts"), { bolts: "Further angle, only if your source gives one", check: true }] }],
  },
  {
    id: "piston-back",
    group: "install",
    title: "Wind the piston back",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    when: { parking: "lever" },
    say: "On a cable parking brake caliper the piston screws back: turn it clockwise with the wind-back tool while pressing gently, until it is fully home. Watch the brake fluid reservoir level as you go.",
    tasks: [{ type: "check", items: ["Reservoir checked: some fluid drawn off if it was at MAX", "Piston wound fully back, dust boot still seated", "Piston slot lined up as your source shows for the pad's pin, if it has one"] }],
  },
  {
    id: "pads-caliper-on",
    group: "install",
    title: "New pads and the caliper",
    figure: "corner",
    focus: ["innerPad", "outerPad", "caliper"],
    install: ["innerPad", "outerPad", "caliper"],
    ref: "",
    say: "Fit the new pads to the carrier, with paste only where your source allows. Lower the caliper over them and torque its bolts.",
    tasks: [{ type: "check", items: ["Pads in, friction surfaces clean"] }, { type: "torque", what: "Caliper bolts", stages: [pending("Caliper bolts")] }],
  },
  {
    id: "finish",
    group: "finish",
    title: "Close service mode, wheels on, test",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    say: "Do the other side the same way. With both sides done, end service mode so the motors bring the pads up to the discs, or with a cable parking brake, pump the pedal and set the parking brake as your source says. Then the wheels go on.",
    tasks: [
      { type: "check", when: { parking: "electric" }, items: ["Both sides done", "Service mode ended with the diagnostic tool; parking brake applied and released once to check", "No parking brake warning on the dashboard"] },
      { type: "check", when: { parking: "lever" }, items: ["Both sides done", "Brake pedal pumped until firm", "Parking brake travel checked and adjusted only if your source says so"] },
      { type: "order", diagram: "wheel", q: "Tighten the wheel bolts in a star pattern, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "torque wrench", stages: [pending("All five, in the star pattern")] },
      { type: "check", items: ["Brake fluid level between MIN and MAX", "Brakes tested at low speed, then bedded in gently", "Parking brake holds the car on a slope"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: { wheel: wheelStar() }, figures: { corner: corner("A rear brake corner seen from outboard: the hub, the brake disc, the caliper carrier, the inner and outer pads, the caliper and the wheel. The parking brake motor on the caliper is not drawn.") } };
}
