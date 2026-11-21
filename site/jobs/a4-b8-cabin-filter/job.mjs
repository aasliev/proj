// Audi A4 (B8) cabin (pollen) filter: a DRAFT job.
import { draftMeta } from "../../lib/a4b8.mjs";
import { filterBox } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-cabin-filter",
  title: "Cabin (pollen) filter",
  system: "Service",
  minutes: 20,
  difficulty: "Beginner",
  summary: "Replace the filter that cleans the air coming into the cabin: fewer smells, less dust and pollen, and windows that demist better.",
});

const VARIANTS = {
  type: {
    q: "Filter type",
    help: "A plain filter is white paper; an activated carbon filter is grey or black and also takes out smells and exhaust fumes. Either fits if it is the right part for the car.",
    options: [
      { id: "plain", label: "Plain (pollen)" },
      { id: "carbon", label: "Activated carbon" },
    ],
  },
};

const GROUPS = [{ id: "job", name: "Cabin filter" }];
const STEPS = [
  {
    id: "find",
    group: "job",
    title: "Find the filter and get the right part",
    figure: "cabin",
    focus: ["lid"],
    ref: "",
    say: "The filter sits in the air intake for the heater and air conditioning. Where depends on the car: commonly in the passenger footwell behind a cover, behind the glovebox, or under the panel below the windscreen. Your source or the owner's manual shows where it is on yours.",
    tasks: [{ type: "variant", keys: ["type"] }, { type: "check", items: ["Location and access panel found in your source (spec not loaded)", "New filter for this car", "Torch, and screwdriver or Torx bits for the cover"] }],
  },
  {
    id: "open",
    group: "job",
    title: "Remove the cover",
    figure: "cabin",
    focus: ["lid"],
    remove: ["lid"],
    ref: "",
    say: "Switch the ignition and the blower off. Remove the access panel and the filter cover. Plastic clips break easily when cold: work slowly.",
    tasks: [{ type: "check", items: ["Ignition and blower off", "Access panel and filter cover off, screws kept together"] }],
  },
  {
    id: "swap",
    group: "job",
    title: "Swap the filter",
    figure: "cabin",
    focus: ["element"],
    remove: ["element"],
    ref: "",
    say: "Slide the old filter out slowly so the dirt stays on it. Note the airflow arrow on its edge. Vacuum leaves and dust out of the housing.",
    tasks: [
      { type: "check", items: ["Old filter out, airflow arrow direction noted", "Housing vacuumed clean"] },
      {
        type: "choice",
        q: "Which way should the arrow on the new filter point?",
        options: [
          { text: "Whichever way it fits", ok: false, why: "Many filters fit both ways, but only one way works as designed." },
          { text: "In the direction the air flows, as on the old filter or the housing", ok: true, why: "The arrow shows airflow. Facing it the wrong way reduces filtering, and for a carbon filter it reduces smell removal." },
          { text: "Always towards the windscreen", ok: false, why: "It depends on how the housing is built: follow the air, not the windscreen." },
        ],
      },
    ],
  },
  {
    id: "fit",
    group: "job",
    title: "Fit the new filter and the cover",
    figure: "cabin",
    focus: ["element", "lid"],
    install: ["element", "lid"],
    ref: "",
    say: "Slide the new filter in with its arrow in the airflow direction, without folding its edges. Refit the cover and the panel, then run the blower at full to check the airflow and listen for rattles.",
    tasks: [{ type: "check", items: ["New filter in, arrow in the airflow direction, edges not folded", "Cover and panel refitted", "Blower tested at every speed", "Date and mileage noted"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: {}, diagrams: {}, figures: { cabin: filterBox({ id: "cabin", flat: true, label: "A cabin filter tray: the housing, the flat pleated filter, and the cover with its catch." }) } };
}
