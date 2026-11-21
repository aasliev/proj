// Jump-starting the Audi A4 (B8): a DRAFT job. The connection order is general practice;
// the car's jump-start points come from the owner's manual.
import { draftMeta } from "../../lib/a4b8.mjs";
import { battery } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-jump-start",
  title: "Jump start",
  system: "Body and electrics",
  minutes: 20,
  difficulty: "Beginner",
  summary: "Start a car with a flat battery from another car or a jump pack: the right connection order, the jump-start points, and what to do afterwards.",
  draftNote: "The connection order is general practice. Where the jump-start points are on your car is in the owner's manual: on many cars they are under the hood, even when the battery is elsewhere.",
});

const VARIANTS = {
  donor: {
    q: "What will you start it from?",
    help: "Another car with a 12 V battery and jumper cables, or a portable jump pack.",
    options: [
      { id: "car", label: "Another car" },
      { id: "pack", label: "Jump pack" },
    ],
  },
};

const GROUPS = [{ id: "prep", name: "Before connecting" }, { id: "jump", name: "Jump" }, { id: "after", name: "Afterwards" }];

const STEPS = [
  {
    id: "safe",
    group: "prep",
    title: "Check it's safe to jump",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "Never jump a battery that is cracked, leaking or frozen: it can explode. Batteries give off hydrogen: no smoking or sparks near them. Use the jump-start points the owner's manual shows rather than the battery itself if it has them.",
    tasks: [
      { type: "variant", keys: ["donor"] },
      { type: "check", items: ["Battery not cracked, leaking, bulging or frozen", "Jump-start points found in the owner's manual", "Both cars off, all electrical loads off (lights, heater blower, radio)", "Cars not touching each other"] },
      {
        type: "choice",
        q: "The dead car's battery case is bulging. What do you do?",
        options: [
          { text: "Jump it carefully", ok: false, why: "A bulging battery has failed inside and can burst or explode." },
          { text: "Don't jump it: replace the battery", ok: true, why: "A swollen, cracked or frozen battery must not be charged or jumped." },
          { text: "Charge it slowly instead", ok: false, why: "Charging a swollen battery is just as dangerous." },
        ],
      },
    ],
  },
  {
    id: "red",
    group: "jump",
    title: "Red cable: positive to positive",
    figure: "battery",
    focus: ["posClamp"],
    ref: "",
    say: "Connect the red cable to the dead car's positive (+) jump point first, then the other end to the donor's positive (+). Keep the free clamps apart and away from metal.",
    tasks: [{ type: "check", items: ["Red to the dead car's +", "Red to the donor's + (or the pack's red clamp is the one fitted)"] }],
  },
  {
    id: "black",
    group: "jump",
    title: "Black cable: donor negative, then a ground point",
    figure: "battery",
    focus: ["negClamp"],
    ref: "",
    say: "Connect the black cable to the donor's negative (−), then the last clamp to the dead car's ground point: a bare metal point on the engine or body that the owner's manual shows, away from the battery. Any spark from the last connection then happens away from battery gases.",
    tasks: [
      { type: "check", items: ["Black to the donor's −", "Last clamp to the dead car's ground point, not to the battery's − post"] },
      {
        type: "choice",
        q: "Why does the last clamp go on a ground point away from the battery?",
        options: [
          { text: "The last connection can spark, and a battery can give off explosive hydrogen", ok: true, why: "Making the final connection away from the battery keeps the spark away from the gas." },
          { text: "It gives a stronger current", ok: false, why: "The current is the same; it's about where the spark happens." },
          { text: "To protect the radio", ok: false, why: "It's about the spark, not the radio." },
        ],
      },
    ],
  },
  {
    id: "start",
    group: "jump",
    title: "Start",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "With another car: start the donor and let it run for a few minutes. Then try the dead car for no more than about 10 seconds at a time, with a pause between tries. If it won't start after a few tries, stop: something else is wrong.",
    tasks: [
      { type: "check", when: { donor: "car" }, items: ["Donor running for a few minutes first"] },
      { type: "check", when: { donor: "pack" }, items: ["Pack switched on as its instructions say"] },
      { type: "check", items: ["Dead car started, tries kept short with pauses"] },
    ],
  },
  {
    id: "disconnect",
    group: "after",
    title: "Disconnect in reverse order",
    figure: "battery",
    focus: ["negClamp", "posClamp"],
    ref: "",
    say: "Leave the jumped car running. Remove the clamps in the reverse order: ground point, donor −, donor +, jumped car +. Don't let the clamps touch each other or the car.",
    tasks: [{ type: "check", items: ["Ground point clamp off first", "Donor − off", "Donor + off", "Jumped car + off last; jump point covers closed"] }],
  },
  {
    id: "after",
    group: "after",
    title: "Afterwards",
    figure: "battery",
    focus: ["battery"],
    ref: "",
    say: "Drive for at least half an hour, or charge the battery with a charger: a short trip won't recharge it. Then find out why it went flat: test the battery (see the battery job), and think about lights left on or a fault draining it.",
    tasks: [{ type: "check", items: ["Car driven for a good while, or the battery charged", "Battery tested once charged (see the battery job)", "Cause looked for: lights left on, short trips only, an old battery, or a drain"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { battery: battery() } };
}
