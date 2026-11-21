// Audi A4 (B8) headlamp bulb replacement: a DRAFT job, halogen headlamps only.
import { draftMeta } from "../../lib/a4b8.mjs";
import { headlamp } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-bulbs",
  title: "Headlamp bulb replacement",
  system: "Body and electrics",
  minutes: 30,
  difficulty: "Beginner",
  summary: "Replace a halogen headlamp bulb from behind the lamp, without touching the glass, and check the beam afterwards. Xenon and LED lamps are flagged as shop jobs.",
});

const VARIANTS = {
  lamp: {
    q: "Headlamp type",
    help: "Look at the lamp and the owner's manual. Xenon (HID) lamps give a bluish-white light that takes a second to come up to full; LED lamps have rows of small LEDs; halogen bulbs light at once with a warmer light.",
    options: [
      { id: "halogen", label: "Halogen" },
      { id: "xenon", label: "Xenon (HID)" },
      { id: "led", label: "LED" },
    ],
  },
};

const GROUPS = [{ id: "job", name: "Headlamp bulb" }];
const STEPS = [
  {
    id: "type",
    group: "job",
    title: "Identify the lamp and the bulb",
    figure: "headlamp",
    focus: ["housing"],
    ref: "",
    say: "Bulb types differ between dipped beam, main beam and the other lamps. The owner's manual lists them, and how you reach the back of the lamp: from the engine bay, or through the wheel arch liner.",
    tasks: [
      { type: "variant", keys: ["lamp"] },
      { type: "note", when: { lamp: "xenon" }, text: "Stop: xenon lamps run at very high voltage, enough to injure or kill even when switched off. Follow your source's procedure exactly, or have a workshop do it." },
      { type: "note", when: { lamp: "led" }, text: "LED headlamps usually have no replaceable bulb: a failed LED means a module or lamp unit, which is a workshop job. Follow your source." },
      { type: "check", when: { lamp: "halogen" }, items: ["Bulb type for this lamp from the owner's manual (spec not loaded)", "New bulb, ideally replace both sides together", "Clean gloves or a cloth to hold the bulb"] },
    ],
  },
  {
    id: "cover",
    group: "job",
    title: "Remove the dust cover",
    figure: "headlamp",
    focus: ["cover"],
    remove: ["cover"],
    ref: "",
    when: { lamp: "halogen" },
    say: "Lights off and key out. Reach the back of the lamp as your source describes, then turn or unclip the round dust cover over the bulb.",
    tasks: [{ type: "check", items: ["Lights off, key out, lamp cool", "Access opened (engine bay or wheel arch liner) as your source describes", "Dust cover off"] }],
  },
  {
    id: "out",
    group: "job",
    title: "Take the old bulb out",
    figure: "headlamp",
    focus: ["connector", "bulb"],
    remove: ["connector", "bulb"],
    ref: "",
    when: { lamp: "halogen" },
    say: "Pull the connector off the bulb. Release the bulb (a wire spring clip, or a twist of the bulb holder) and lift it out, noting how its tabs sit.",
    tasks: [{ type: "check", items: ["Connector off", "Retaining clip or holder released", "Old bulb out, tab position noted"] }],
  },
  {
    id: "in",
    group: "job",
    title: "Fit the new bulb",
    figure: "headlamp",
    focus: ["bulb", "connector"],
    install: ["bulb", "connector"],
    ref: "",
    when: { lamp: "halogen" },
    say: "Hold the new bulb by its metal base only. Seat it with the tabs in their slots, lock the clip or holder, and push the connector on.",
    tasks: [
      { type: "check", items: ["Bulb held by the base only", "Tabs in their slots, clip or holder locked", "Connector pushed fully on"] },
      {
        type: "choice",
        q: "You touched the bulb's glass with your fingers. What now?",
        options: [
          { text: "Nothing, it's fine", ok: false, why: "Skin oil on halogen glass creates a hot spot that shortens the bulb's life or cracks it." },
          { text: "Clean the glass with a cloth and alcohol before fitting", ok: true, why: "Clean it off, let it dry, and fit it holding only the base." },
          { text: "Wipe it on your shirt", ok: false, why: "That smears the oil rather than removing it." },
        ],
      },
    ],
  },
  {
    id: "close",
    group: "job",
    title: "Refit the cover and check the beam",
    figure: "headlamp",
    focus: ["cover"],
    install: ["cover"],
    ref: "",
    when: { lamp: "halogen" },
    say: "Refit the dust cover fully, or water and dirt get into the lamp. Switch on and check the beam against a wall: both sides at the same height. If it's clearly off, have the aim set at a garage.",
    tasks: [{ type: "check", items: ["Dust cover seated all round", "Access panels refitted", "Lamp works on dipped and main beam", "Beam height looks the same both sides; aim checked if in doubt"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { headlamp: headlamp() } };
}
