// Audi A4 (B8) battery test and replacement: a DRAFT job. The resting-voltage guide is general
// lead-acid practice; the battery type, location, torques and registration come from licensed data.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { battery } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-battery",
  title: "Battery test and replacement",
  system: "Body and electrics",
  minutes: 45,
  difficulty: "Beginner–intermediate",
  summary: "Test the battery's resting voltage, then swap it safely: negative off first and on last, the right battery type, and the new battery registered with the car.",
  draftNote: "The procedure is general workshop practice, and the voltage guide is general for 12 V lead-acid batteries. No licensed data is loaded yet, so the battery location, type, terminal torques and the registration procedure show as “spec not loaded”.",
});

const VARIANTS = {
  type: {
    q: "Battery type (printed on the old battery)",
    help: "Cars with start-stop usually have an AGM battery; others may have an EFB or a standard flooded battery. Fit the same type, or the one your source specifies: the charging system is set for it.",
    options: [
      { id: "agm", label: "AGM" },
      { id: "efb", label: "EFB" },
      { id: "std", label: "Standard (flooded)" },
    ],
  },
};

// General guide for a 12 V lead-acid battery that has rested for several hours.
const SPEC = { voltage: { min: 12.4, max: 12.9, unit: "V", ref: "general practice for 12 V lead-acid batteries, rested" } };

const GROUPS = [{ id: "test", name: "Test" }, { id: "swap", name: "Replace" }, { id: "after", name: "After the swap" }];
const STEPS = [
  {
    id: "test",
    group: "test",
    title: "Test the resting voltage",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "Leave the car off for a few hours (overnight is best) so you read the battery's real state of charge, not a surface charge from driving. Measure across the terminals with a multimeter set to DC volts. Around 12.6 V or more is full; below 12.4 V it needs charging; below about 12 V it is flat.",
    tasks: [
      { type: "measure", spec: "voltage", label: "Resting voltage across the terminals", items: ["Reading"] },
      { type: "check", items: ["If low: battery charged with a suitable charger and re-tested the next day", "If it won't hold a charge, or the car starts slowly even when charged: battery replaced (a load test at a parts shop confirms it)"] },
    ],
  },
  {
    id: "prep",
    group: "swap",
    title: "Before you disconnect",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "Disconnecting the battery can reset the radio, clock, window one-touch settings and stored faults. Have the radio code if your car needs one. Battery acid burns and a shorted battery can spark: wear eye protection and remove rings and watches.",
    tasks: [
      { type: "variant", keys: ["type"] },
      { type: "check", items: ["New battery of the same type and size, and capacity (Ah) and cold-cranking rating (A) as your source specifies (spec not loaded)", "Battery location and any covers found in your source (spec not loaded)", "Radio code to hand if needed", "Ignition off, key out of the car, all doors closed and locks working"] },
    ],
  },
  {
    id: "neg-off",
    group: "swap",
    title: "Disconnect the negative terminal first",
    figure: "battery",
    focus: ["negClamp"],
    remove: ["negClamp"],
    ref: "",
    say: "Undo the negative (−) clamp first and move it well away from the post. Then, if a spanner touches the car's metal while you're on the positive, nothing happens: the circuit is already broken.",
    tasks: [
      { type: "check", items: ["Negative clamp off and tucked away so it can't spring back"] },
      {
        type: "choice",
        q: "Why does the negative come off first?",
        options: [
          { text: "It's the larger terminal", ok: false, why: "Size isn't the reason." },
          { text: "The car's body is connected to negative, so a spanner touching metal while on the positive would short the battery", ok: true, why: "With the negative off first, an accidental contact can't complete a circuit." },
          { text: "To protect the radio", ok: false, why: "The order is about shorts and sparks, not the radio." },
        ],
      },
    ],
  },
  {
    id: "pos-off",
    group: "swap",
    title: "Disconnect the positive and take the battery out",
    figure: "battery",
    focus: ["posClamp", "holdDown", "battery"],
    remove: ["posClamp", "holdDown", "battery"],
    ref: "",
    say: "Lift the red cover and undo the positive (+) clamp. Unplug the vent hose if there is one, undo the hold-down clamp and lift the battery out upright: it's heavy, so use the handle and lift with your legs.",
    tasks: [{ type: "check", items: ["Positive clamp off", "Vent hose unplugged, if fitted", "Hold-down undone", "Battery lifted out upright", "Tray and terminals cleaned; any white or green deposits removed"] }],
  },
  {
    id: "in",
    group: "swap",
    title: "Fit the new battery and clamp it down",
    figure: "battery",
    focus: ["battery", "holdDown"],
    install: ["battery", "holdDown"],
    ref: "",
    say: "Set the new battery in the same way round, refit the vent hose and fasten the hold-down. A loose battery can shake itself apart.",
    tasks: [
      { type: "check", items: ["Battery in, terminals the same side as before", "Vent hose fitted, if there is one"] },
      { type: "torque", what: "Hold-down clamp", stages: [pending("Hold-down bolt")] },
    ],
  },
  {
    id: "connect",
    group: "swap",
    title: "Connect positive first, negative last",
    figure: "battery",
    focus: ["posClamp", "negClamp"],
    install: ["posClamp", "negClamp"],
    ref: "",
    say: "Positive on first, then negative: the reverse of removal, for the same reason. Tighten the clamps so they can't be turned by hand on the posts, without overtightening and cracking them.",
    tasks: [{ type: "torque", what: "Terminal clamps", stages: [pending("Positive clamp"), pending("Negative clamp, last")] }, { type: "check", items: ["Red cover over the positive"] }],
  },
  {
    id: "register",
    group: "after",
    title: "Register the battery and reset",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "Many modern Audis manage charging by the battery's age and type, and need a new battery registered with a diagnostic tool, or they keep charging it as if it were the old one. Check your source for this car.",
    tasks: [{ type: "check", items: ["New battery registered with a diagnostic tool, if your source requires it", "Clock and radio set", "Windows' one-touch function relearned, as the owner's manual describes", "Old battery returned to the seller or a recycling point"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: SPEC, diagrams: {}, figures: { battery: battery() } };
}
