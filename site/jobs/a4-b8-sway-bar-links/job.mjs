// Audi A4 (B8) anti-roll bar (sway bar) links: a DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { suspension } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-sway-bar-links",
  title: "Anti-roll bar (sway bar) links",
  system: "Suspension and steering",
  minutes: 60,
  difficulty: "Beginner–intermediate",
  summary: "A rattle or light clunk over small bumps is often a worn anti-roll bar link. Check for play, then swap the link, holding the ball stud so it doesn't spin.",
});

const GROUPS = [{ id: "check", name: "Check" }, { id: "swap", name: "Replace" }];

const STEPS = [
  {
    id: "check",
    group: "check",
    title: "Find a worn link",
    figure: "suspension",
    focus: ["swayLink", "swayBar"],
    ref: "",
    say: "The link joins the anti-roll bar to the suspension, with a small ball joint at each end. A worn one rattles or clunks over small bumps and rough roads, often more at low speed.",
    tasks: [
      { type: "check", items: ["Car on its wheels or on ramps, so the bar isn't loaded to one side", "Link grabbed and shaken by hand: any knock or free play at its ball joints means replace", "Rubber boots on the joints checked for tears"] },
      {
        type: "choice",
        q: "Why check the link with the car on its wheels, not hanging?",
        options: [
          { text: "With one wheel hanging, the bar twists and loads the link, hiding play", ok: true, why: "With both sides at the same height the link is unloaded and play is easy to feel." },
          { text: "It's safer", ok: false, why: "Safety matters, but the reason here is the load on the link." },
          { text: "The link can only be reached from above", ok: false, why: "It's reached from the wheel house either way." },
        ],
      },
    ],
  },
  {
    id: "remove",
    group: "swap",
    title: "Remove the link",
    figure: "suspension",
    focus: ["swayLink"],
    remove: ["swayLink"],
    ref: "",
    say: "With the wheel off (or turned for access), undo the nut at each end. The ball stud will spin with the nut: hold it with a hex or Torx key in the end of the stud, or a wrench on its flats. Penetrating oil the day before helps with rusted nuts.",
    tasks: [{ type: "check", items: ["Car safely supported; wheel off if needed for access", "Both nuts undone while holding the stud", "Link out"] }],
  },
  {
    id: "install",
    group: "swap",
    title: "Fit the new link",
    figure: "suspension",
    focus: ["swayLink"],
    install: ["swayLink"],
    ref: "",
    say: "Fit the new link with new self-locking nuts, holding the stud while tightening. If the bar is twisted and the holes don't line up, lift the other side's suspension slightly with the jack or do it with both wheels at the same height.",
    tasks: [
      { type: "torque", what: "Link nuts (new, self-locking)", stages: [pending("Upper nut"), pending("Lower nut")] },
      { type: "check", items: ["Boots not twisted or pinched", "Wheel refitted and torqued (see the wheel swap job)", "Test drive over small bumps: rattle gone"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { suspension: suspension() } };
}
