// Audi A4 (B8) flat tire at the roadside: spare wheel or tire repair kit. A DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { corner } from "../a4-b8-front-brakes/figures.mjs";
import { wheelStar } from "../../lib/diagrams.mjs";

export const meta = draftMeta({
  id: "a4-b8-flat-tire",
  title: "Flat tire: spare wheel or repair kit",
  system: "Wheels and tires",
  minutes: 40,
  difficulty: "Beginner",
  summary: "Deal with a puncture safely at the roadside: fit the spare wheel with the car's jack, or seal the tire with the repair kit, and drive on within the limits.",
  draftNote: "The procedure is general practice. No licensed data is loaded yet, so the wheel bolt torque shows as “spec not loaded”. The jack points, the space-saver's speed limit and the repair kit's instructions are in the owner's manual and on the kit.",
});

const VARIANTS = {
  kit: {
    q: "What does your car carry?",
    help: "Look under the trunk floor: a spare wheel (full size or a narrow space-saver) with a jack, or a tire repair kit (sealant bottle and compressor).",
    options: [
      { id: "spare", label: "Spare wheel and jack" },
      { id: "repair", label: "Repair kit" },
    ],
  },
};

const GROUPS = [{ id: "safe", name: "Get safe" }, { id: "fix", name: "Fix" }, { id: "after", name: "Afterwards" }];

const STEPS = [
  {
    id: "safe",
    group: "safe",
    title: "Get the car and people safe",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "Pull over as far from traffic as you can, on firm level ground. Hazard lights on, everyone out on the side away from traffic and behind a barrier if there is one. On a highway or a narrow road, call for help instead of working beside traffic.",
    tasks: [
      { type: "variant", keys: ["kit"] },
      { type: "check", items: ["Stopped on firm, level ground away from traffic", "Hazard lights on, parking brake on, transmission in P (or first gear for a manual)", "Warning triangles or flares set out, if you carry them", "Reflective vest on, if you have one"] },
    ],
  },
  {
    id: "repair-kit",
    group: "fix",
    title: "Seal the tire with the repair kit",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    when: { kit: "repair" },
    say: "The kit works for small punctures in the tread, not for cuts in the sidewall or a tire that's come off the rim. Follow the kit's own instructions: shake the sealant, fill through the valve, inflate with the compressor, then drive a short distance to spread the sealant and check the pressure again.",
    tasks: [
      { type: "check", items: ["Damage is a small hole in the tread, not the sidewall", "Sealant filled and tire inflated as the kit's instructions say", "Speed limit sticker from the kit stuck where the driver can see it", "Pressure rechecked after the short drive the kit specifies"] },
      {
        type: "choice",
        q: "The tire has a 3 cm cut in its sidewall. Can the kit fix it?",
        options: [
          { text: "Yes, sealant fills any hole", ok: false, why: "Sidewall damage can't be sealed and the tire can fail suddenly." },
          { text: "No: fit a spare or call for help", ok: true, why: "Repair kits are only for small tread punctures." },
          { text: "Yes, if you drive slowly", ok: false, why: "Speed doesn't make a sidewall cut safe." },
        ],
      },
    ],
  },
  {
    id: "loosen",
    group: "fix",
    title: "Loosen the bolts and jack up",
    figure: "corner",
    focus: ["bolts", "wheel"],
    remove: ["bolts", "wheel"],
    ref: "",
    when: { kit: "spare" },
    say: "Take out the spare, jack and wheel brace. Slacken each bolt half a turn while the wheel is on the ground. Put the jack only at the jack point the owner's manual shows for that corner, raise the car until the tire clears the ground, then take out the bolts and the wheel. Never put any part of your body under the car.",
    tasks: [{ type: "check", items: ["Spare, jack, brace and locking bolt adapter out", "Bolts slackened on the ground", "Jack at the correct jack point, on firm ground", "Car raised just enough; bolts out, wheel off and laid flat under the rocker panel as a safety catch"] }],
  },
  {
    id: "fit",
    group: "fix",
    title: "Fit the spare and tighten",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    when: { kit: "spare" },
    say: "Lift the spare onto the hub, screw in all the bolts by hand, and snug them up evenly. Lower the car and tighten in a star pattern with the brace. Have the torque checked with a torque wrench as soon as you can.",
    tasks: [
      { type: "order", diagram: "wheel", q: "Tighten in a star pattern, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "wheel brace now, torque wrench soon after", stages: [pending("All five, in the star pattern")] },
    ],
  },
  {
    id: "after",
    group: "after",
    title: "Drive on within the limits",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "A space-saver spare and a sealed tire both have a speed limit and are temporary. Drive gently to a tire shop, and tell them if you used sealant.",
    tasks: [
      { type: "check", when: { kit: "spare" }, items: ["Spare's pressure checked (a space-saver usually needs a high pressure, printed on it)", "Speed limit for a space-saver respected, as printed on the wheel or in the owner's manual", "Wheel bolt torque checked with a torque wrench", "Flat tire and tools stowed"] },
      { type: "check", when: { kit: "repair" }, items: ["Speed limit from the kit respected", "Tire shop told that sealant was used", "New sealant bottle bought for the kit"] },
      { type: "check", items: ["Tire pressure monitor reset, if your car needs it after a wheel or pressure change"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: { wheel: wheelStar() }, figures: { corner: corner("A wheel and brake corner seen from outboard: the hub, the disc, the brake and the wheel with its five bolts.") } };
}
