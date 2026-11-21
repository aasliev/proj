// Audi A4 (B8) wiper blades: a DRAFT job.
import { draftMeta } from "../../lib/a4b8.mjs";
import { wiper } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-wipers",
  title: "Wiper blades",
  system: "Body and electrics",
  minutes: 15,
  difficulty: "Beginner",
  summary: "Put the wipers in their service position, swap the front blades without the arm snapping onto the glass, and check the rear blade.",
});

const VARIANTS = {
  rear: {
    q: "Rear wiper",
    help: "Saloons have no rear wiper; Avant (estate) and allroad models do.",
    options: [
      { id: "yes", label: "Rear wiper fitted" },
      { id: "no", label: "No rear wiper" },
    ],
  },
};

const GROUPS = [{ id: "front", name: "Front blades" }, { id: "finish", name: "Rear and test" }];
const STEPS = [
  {
    id: "parts",
    group: "front",
    title: "Get the right blades",
    figure: "wiper",
    focus: ["blade"],
    ref: "",
    say: "Driver and passenger blades are usually different lengths, and the arm's fitting decides which blades fit. Buy by car model and year, or compare with the old blades.",
    tasks: [{ type: "variant", keys: ["rear"] }, { type: "check", items: ["Front blades for this car, both lengths right", "Rear blade, if fitted", "A towel to protect the windscreen"] }],
  },
  {
    id: "service",
    group: "front",
    title: "Put the wipers in the service position",
    figure: "wiper",
    focus: ["arm"],
    ref: "",
    say: "Parked, the arms sit under the edge of the bonnet and can't be lifted. Many Audis have a service position: switch the ignition on and off, then push the wiper stalk down briefly, and the wipers stop upright on the glass. The owner's manual describes it for your car.",
    tasks: [
      { type: "check", items: ["Wipers moved to the service position as your owner's manual describes", "Towel laid on the windscreen under the arm"] },
      {
        type: "choice",
        q: "Why lay a towel on the windscreen?",
        options: [
          { text: "To keep the glass clean", ok: false, why: "That's not the risk." },
          { text: "A bare arm can spring back and crack the windscreen", ok: true, why: "Without a blade, the arm's spring slams the metal end onto the glass." },
          { text: "To stop water dripping into the car", ok: false, why: "No water is involved." },
        ],
      },
    ],
  },
  {
    id: "off",
    group: "front",
    title: "Take the old blade off",
    figure: "wiper",
    focus: ["blade"],
    remove: ["blade"],
    ref: "",
    say: "Fold the arm away from the glass, holding it. Press the release tab where the blade joins the arm and slide the blade off along the arm. Lower the arm onto the towel.",
    tasks: [{ type: "check", items: ["Arm lifted and held", "Release tab pressed, blade slid off", "Arm lowered gently onto the towel"] }],
  },
  {
    id: "on",
    group: "front",
    title: "Fit the new blade",
    figure: "wiper",
    focus: ["blade"],
    install: ["blade"],
    ref: "",
    say: "Slide the new blade onto the arm until it clicks, and pull on it to make sure it's locked. Do both sides, then switch the ignition on and work the stalk to return the wipers to their parked position.",
    tasks: [{ type: "check", items: ["Driver's side blade clicked in and pulled to check", "Passenger's side blade clicked in and pulled to check", "Wipers returned to their parked position"] }],
  },
  {
    id: "rear",
    group: "finish",
    title: "Rear blade",
    figure: "wiper",
    focus: ["blade"],
    ref: "",
    when: { rear: "yes" },
    say: "Lift the rear arm, unclip the blade as its fitting allows, and clip on the new one. Lower the arm gently onto the glass.",
    tasks: [{ type: "check", items: ["Rear blade swapped and locked", "Arm lowered gently"] }],
  },
  {
    id: "test",
    group: "finish",
    title: "Test",
    figure: "wiper",
    focus: ["arm", "blade"],
    ref: "",
    say: "Wet the screen with the washers and run the wipers at each speed. New blades should leave the glass clear without streaks or judder.",
    tasks: [{ type: "check", items: ["Washers and wipers tested at each speed", "No streaks, judder or noise", "Arms park in their normal place"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { wiper: wiper() } };
}
