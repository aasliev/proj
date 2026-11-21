// Audi A4 (B8) auxiliary (ribbed) belt: a DRAFT job.
import { draftMeta } from "../../lib/a4b8.mjs";
import { beltDrive } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-aux-belt",
  title: "Auxiliary belt",
  system: "Engine",
  minutes: 60,
  difficulty: "Intermediate",
  summary: "Check the ribbed belt that drives the alternator and air conditioning, and replace it: draw the routing first, release the tensioner, fit the new belt in every groove.",
});

const GROUPS = [{ id: "check", name: "Check" }, { id: "swap", name: "Replace" }];

const STEPS = [
  {
    id: "inspect",
    group: "check",
    title: "Check the belt",
    figure: "belt",
    focus: ["beltLoop"],
    ref: "",
    say: "With the engine off and the key out, look at the belt along its whole length, turning the engine by hand at the crankshaft bolt if your source allows. A squeal on cold starts or with the air conditioning on can mean a worn belt or a weak tensioner.",
    tasks: [
      { type: "check", items: ["Engine off, key out of the car", "Ribs checked for cracks across them, missing chunks, fraying and glazing", "Tensioner holds the belt firmly; its pulley turns smoothly and quietly", "Pulleys checked for wobble and noise"] },
      {
        type: "choice",
        q: "The belt has a few fine cracks across some ribs. Is it still OK?",
        options: [
          { text: "A few fine cracks are normal aging; many cracks or missing pieces mean replace it", ok: true, why: "Replace it if cracks are many and close together, pieces are missing, or it's at the interval your source gives." },
          { text: "Any crack means the belt is about to snap", ok: false, why: "Fine cracks appear with age; judge by how many and how deep." },
          { text: "Cracks don't matter as long as it doesn't squeal", ok: false, why: "A belt can fail without warning: look, don't only listen." },
        ],
      },
    ],
  },
  {
    id: "route",
    group: "swap",
    title: "Record the routing",
    figure: "belt",
    focus: ["beltLoop", "pulleys"],
    ref: "",
    say: "Before anything comes off, photograph the belt or find the routing diagram on its label or in your source. Ribbed sides run on grooved pulleys; the flat back runs on smooth idler pulleys.",
    tasks: [{ type: "check", items: ["Routing photographed or found in your source", "Access gained as your source describes (covers, and anything in the way) (spec not loaded)", "New belt of the right length and rib count"] }],
  },
  {
    id: "off",
    group: "swap",
    title: "Release the tensioner and take the belt off",
    figure: "belt",
    focus: ["tensioner", "beltLoop"],
    remove: ["beltLoop"],
    ref: "",
    say: "Turn the tensioner with a wrench or belt tool in the direction your source shows to take the load off the belt, lock it if it has a locking hole, and slip the belt off one smooth pulley. Let the tensioner back gently: a spring-loaded tensioner that snaps back can be damaged.",
    tasks: [{ type: "check", items: ["Tensioner turned in the right direction and locked with a pin, if it has a locking hole", "Belt off", "Tensioner and idler pulleys spun by hand: smooth and quiet, no wobble"] }],
  },
  {
    id: "on",
    group: "swap",
    title: "Fit the new belt",
    figure: "belt",
    focus: ["beltLoop"],
    install: ["beltLoop"],
    ref: "",
    say: "Lay the belt round every pulley following your photo, leaving a smooth pulley until last. Release the tensioner onto it. Check every rib is in its groove: a belt one rib off shreds quickly.",
    tasks: [
      { type: "check", items: ["Belt routed as in the photo or diagram", "Tensioner released gently onto the belt, locking pin removed", "Every rib in its groove on every grooved pulley", "Covers and anything removed for access refitted"] },
      {
        type: "choice",
        q: "After fitting, the belt sits half off the edge of one pulley. What do you do?",
        options: [
          { text: "Start the engine; it will settle", ok: false, why: "It won't: it will throw the belt or shred it." },
          { text: "Release the tensioner again and seat the belt properly", ok: true, why: "Every rib must sit in a groove before the engine runs." },
          { text: "Leave it if it's only one rib", ok: false, why: "One rib off is enough to destroy the belt." },
        ],
      },
    ],
  },
  {
    id: "test",
    group: "swap",
    title: "Run and check",
    figure: "belt",
    focus: ["beltLoop"],
    ref: "",
    say: "Start the engine, switch on the air conditioning and heated rear window to load the alternator, and watch the belt (from a safe distance, nothing loose near it). Then stop the engine and look again.",
    tasks: [{ type: "check", items: ["Belt runs straight and quiet, also with the air conditioning on", "Battery warning light out", "Belt still in every groove after stopping"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { belt: beltDrive() } };
}
