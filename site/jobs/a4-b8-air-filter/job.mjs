// Audi A4 (B8) engine air filter: a DRAFT job.
import { draftMeta } from "../../lib/a4b8.mjs";
import { filterBox } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-air-filter",
  title: "Engine air filter",
  system: "Service",
  minutes: 20,
  difficulty: "Beginner",
  summary: "Open the air filter housing, swap the element, clean the housing and close it so no unfiltered air can get past.",
});

const GROUPS = [{ id: "job", name: "Air filter" }];
const STEPS = [
  {
    id: "parts",
    group: "job",
    title: "Get the right filter",
    figure: "air",
    focus: ["element"],
    ref: "",
    say: "The air filter cleans the air the engine breathes. The interval is in your service schedule (spec not loaded), sooner in dusty conditions. Order by engine code, since filters differ between engines.",
    tasks: [{ type: "check", items: ["New filter for your engine code", "Screwdriver or Torx bit for the housing screws, if it has them", "Vacuum cleaner or clean rag"] }],
  },
  {
    id: "open",
    group: "job",
    title: "Open the housing",
    figure: "air",
    focus: ["lid"],
    remove: ["lid"],
    ref: "",
    say: "With the engine off, find the black plastic box the intake duct leads to. Undo its clips or screws and lift the lid. If a hose or a sensor plug is on the lid, unplug it first rather than stretching it.",
    tasks: [{ type: "check", items: ["Engine off, key out", "Any plug or hose on the lid disconnected", "Clips or screws undone, lid lifted"] }],
  },
  {
    id: "swap",
    group: "job",
    title: "Swap the element",
    figure: "air",
    focus: ["element"],
    remove: ["element"],
    ref: "",
    say: "Lift out the old element, noting which way up it sat. Clean leaves and dust out of the housing, wiping towards you so nothing drops into the clean-air side leading to the engine.",
    tasks: [
      { type: "check", items: ["Old element out, noted which way up", "Housing cleaned; nothing dropped into the clean side"] },
      {
        type: "choice",
        q: "The old filter looks only slightly dirty. Can you tap it out and refit it?",
        options: [
          { text: "Yes, tapping cleans it", ok: false, why: "Tapping or blowing out damages the paper and opens paths for fine dust. Paper filters are replaced, not cleaned." },
          { text: "Refit it if the interval isn't due; otherwise fit a new one", ok: true, why: "A paper element is replaced at its interval or when it is visibly clogged, never cleaned." },
          { text: "Wash it with water", ok: false, why: "Water ruins a paper element." },
        ],
      },
    ],
  },
  {
    id: "fit",
    group: "job",
    title: "Fit the new element and close up",
    figure: "air",
    focus: ["element", "lid"],
    install: ["element", "lid"],
    ref: "",
    say: "Seat the new element so its rubber edge sits evenly all round the housing. Fit the lid, fasten every clip or screw, and reconnect anything you unplugged. A gap lets dirty air bypass the filter.",
    tasks: [{ type: "check", items: ["Element seated evenly, seal not folded", "Lid closed, every clip or screw fastened", "Plugs and hoses reconnected", "Engine started: no new warning lights", "Date and mileage noted"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { air: filterBox({ id: "air", label: "An engine air filter housing: the lower box with its inlet, the pleated paper element, and the lid with its clips." }) } };
}
