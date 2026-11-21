// Audi A4 (B8) front shock absorbers and springs: a DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { suspension } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-shocks",
  title: "Front shock absorbers",
  system: "Suspension and steering",
  minutes: 300,
  difficulty: "Intermediate–expert",
  summary: "Check the front shocks, then replace them as a pair: take out the strut, compress the spring safely, swap the shock and top mount parts, and refit with the bolts tightened at ride height.",
  draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so torques and the removal steps for your suspension version show as “spec not loaded”. A compressed coil spring stores a lot of energy: use a proper spring compressor in good condition, or have a shop do the spring work.",
});

const GROUPS = [{ id: "check", name: "Check" }, { id: "remove", name: "Remove" }, { id: "rebuild", name: "Swap the shock" }, { id: "install", name: "Install" }];

const STEPS = [
  {
    id: "check",
    group: "check",
    title: "Are the shocks worn?",
    figure: "suspension",
    focus: ["strut"],
    ref: "",
    say: "Worn shocks let the car bounce after bumps, dive under braking and wander on uneven roads. Look for oil on the shock body, and check for knocks from the top mount when turning the steering at standstill.",
    tasks: [
      { type: "check", items: ["Shock body checked for oil leaking down it", "Bounce test: push down hard on a front corner and let go; more than one rebound suggests a worn shock", "Top mount checked for knocking or creaking when turning the wheel at standstill", "Spring checked for a broken coil"] },
      {
        type: "choice",
        q: "One front shock is leaking. What do you replace?",
        options: [
          { text: "Only the leaking one", ok: false, why: "A new and an old shock on one axle make the car brake and corner unevenly." },
          { text: "Both front shocks, as a pair", ok: true, why: "Shocks are always replaced in pairs on an axle, with the top mount parts if they're worn." },
          { text: "All four shocks, always", ok: false, why: "Only if the rear ones are worn too; the rule is pairs per axle." },
        ],
      },
    ],
  },
  {
    id: "prep",
    group: "remove",
    title: "Parts and safety",
    figure: "suspension",
    focus: ["strut"],
    ref: "",
    say: "Get both shocks for your suspension version (standard, sport or adaptive damping, which have different parts), plus new top mount parts and single-use bolts. A spring compressor rated for the spring is essential.",
    tasks: [{ type: "check", items: ["Two shocks for your suspension version (check the PR number on the vehicle data sticker)", "Top mount parts, bump stops and dust boots if worn", "Single-use bolts and nuts from your source (spec not loaded)", "Spring compressor in good condition, rated for the spring", "Car on axle stands, wheel off"] }],
  },
  {
    id: "remove",
    group: "remove",
    title: "Take out the shock and spring assembly",
    figure: "suspension",
    focus: ["strut"],
    remove: ["strut"],
    ref: "",
    say: "Support the wheel bearing housing on a jack. Disconnect the anti-roll bar link and anything clipped to the shock, undo the bolt that holds the shock to its lower mount, then the top mount nuts as your source describes, and lower the assembly out.",
    tasks: [{ type: "check", items: ["Wheel bearing housing supported", "Brake hose, wear wire and sensor wiring unclipped from the shock", "Anti-roll bar link disconnected (see that job)", "Lower mounting bolt out", "Top mount nuts undone; assembly out"] }],
  },
  {
    id: "spring",
    group: "rebuild",
    title: "Compress the spring and swap the shock",
    figure: "suspension",
    focus: ["strut"],
    ref: "",
    say: "Fit the compressor's jaws on opposite sides of the spring, over as many coils as it allows, and compress evenly until the spring is loose between its seats. Only then undo the piston rod nut. Move the top mount, spring and seats to the new shock in the same order and position.",
    tasks: [
      { type: "check", items: ["Compressor jaws seated on opposite sides, safety catches closed", "Spring compressed evenly, loose between its seats", "Piston rod nut undone (hold the rod)", "Parts moved to the new shock in the same order; spring ends against their stops in the seats"] },
      {
        type: "choice",
        q: "The spring is still tight against its seats. Can you undo the piston rod nut?",
        options: [
          { text: "Yes, carefully", ok: false, why: "The nut is all that holds the spring. Released under load, the top mount and spring can fly off and cause serious injury." },
          { text: "No: compress more until the spring is loose between its seats", ok: true, why: "Never undo the nut while the spring is still pressing on its seats." },
        ],
      },
      { type: "torque", what: "Piston rod nut (new)", stages: [pending("Piston rod nut")] },
      { type: "check", items: ["Spring released slowly and evenly; ends against their stops"] },
    ],
  },
  {
    id: "install",
    group: "install",
    title: "Refit and tighten at ride height",
    figure: "suspension",
    focus: ["strut"],
    install: ["strut"],
    ref: "",
    say: "Fit the assembly, top mount nuts first. Fit the lower bolt and refit everything you unclipped. Tighten the bolts through rubber bushings with the suspension at ride height, as for the control arms.",
    tasks: [
      { type: "torque", what: "Top mount nuts (new)", stages: [pending("Top mount nuts")] },
      { type: "check", items: ["Lower mounting bolt fitted; suspension raised to ride height before tightening"] },
      { type: "torque", what: "Lower mounting bolt", stages: [pending("Lower bolt"), { bolts: "Further angle, only if your source gives one", check: true }] },
      { type: "check", items: ["Anti-roll bar link, brake hose and wiring refitted", "Other side done the same way", "Wheels on and torqued (see the wheel swap job)", "Wheel alignment checked and set at a shop", "Adaptive damping, if fitted: basic setting done with a diagnostic tool as your source says"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { suspension: suspension() } };
}
