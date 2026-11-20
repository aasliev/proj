// Audi A4 (B8, 2008–2016) front brake pads and discs: a DRAFT job.
// The procedure is general workshop practice for a single-piston floating caliper. No licensed
// data is loaded yet, so every torque, wear limit and dimension is a `pending` placeholder:
// the page shows "spec not loaded" and grades nothing. Fill SPEC and the torque stages from
// your licensed source, set `ref` on each step to its page, then set status to "ready".
import { buildFigures } from "./figures.mjs";

export const meta = {
  id: "a4-b8-front-brakes",
  title: "Front brake pads and discs",
  vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" },
  system: "Brakes",
  vehicleLine: "Audi A4 (B8, 2008–2016), front axle, single-piston floating caliper",
  status: "draft",
  private: false,
  minutes: 120,
  difficulty: "Beginner–intermediate",
  summary: "Replace the front pads and discs: remove the wheel, caliper, carrier and disc, measure, fit new parts, torque and bed in.",
  source: { name: "Licensed repair data (not loaded yet)", prefix: "Source: " },
  draftNote: "Draft: the procedure is general workshop practice. No licensed data is loaded yet, so torques, wear limits and disc dimensions show as “spec not loaded”. Take every value from your licensed repair data for your car's brake version.",
};

const VARIANTS = {
  caliper: {
    q: "Front caliper type",
    help: "Look at the caliper: a floating caliper has one piston on the inboard side and slides on guide pins; a fixed caliper has pistons on both sides and does not move. Most A4 B8 cars have the floating type; the brake version is on the vehicle data sticker (PR number for the front brake).",
    options: [
      { id: "floating", label: "Floating, one piston" },
      { id: "fixed", label: "Fixed, several pistons" },
    ],
  },
  sensor: {
    q: "Pad wear warning wire",
    help: "Follow the thin wire from the inner pad, if there is one, to its connector on the suspension. Not every car or every axle has one.",
    options: [
      { id: "yes", label: "Wear wire fitted" },
      { id: "no", label: "No wear wire" },
    ],
  },
};

// All pending until licensed data is loaded: the page records readings without grading them.
const SPEC = {};
const pending = (bolts) => ({ bolts, pending: true });

const GROUPS = [
  { id: "prep", name: "Preparation and safety" },
  { id: "remove", name: "Removal" },
  { id: "inspect", name: "Inspection and measuring" },
  { id: "install", name: "Fitting" },
  { id: "finish", name: "Wheel, test and bed-in" },
];

const STEPS = [
  {
    id: "versions",
    group: "prep",
    title: "Identify your brake version",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    say: "Pads, discs and torques differ between brake versions. Identify yours before buying parts: the vehicle data sticker lists the front brake's PR number, and the parts supplier matches parts to it.",
    tasks: [
      { type: "variant", keys: ["caliper", "sensor"] },
      { type: "note", text: "This job covers the floating caliper only. For a fixed multi-piston caliper, stop here and follow the procedure for that caliper in your source.", when: { caliper: "fixed" } },
      { type: "check", items: ["Front brake PR number noted from the vehicle data sticker", "New pads and discs match that PR number (compare the disc diameter and thickness printed on the box)", "New disc retaining screw, and any single-use bolts your source lists"] },
    ],
  },
  {
    id: "safety",
    group: "prep",
    title: "Make the car safe to work on",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    when: { caliper: "floating" },
    say: "Work on level, solid ground. The car's weight goes on axle stands at the jacking points, never on the jack alone. Brake dust can harm your lungs: clean with brake cleaner, never blow it off with compressed air.",
    tasks: [
      { type: "check", items: ["Parking brake on, gear selected (P for automatic)", "Rear wheels chocked", "Wheel bolts slackened half a turn while the wheel is on the ground", "Car raised at the jacking point and set on an axle stand", "Eye protection and gloves on"] },
      {
        type: "choice",
        q: "Where should the car rest while you work under it?",
        options: [
          { text: "On the jack, raised high enough", ok: false, why: "A jack can sink or slip. Never work with any part of your body under a car held only by a jack." },
          { text: "On an axle stand at a proper support point", ok: true, why: "Stands hold the load mechanically; leave the jack lightly touching as a backup if you like." },
          { text: "On the wheel, with the brake removed", ok: false, why: "The wheel has to come off to reach the brake." },
        ],
      },
    ],
  },
  {
    id: "wheel-off",
    group: "remove",
    title: "Take the wheel off",
    figure: "corner",
    focus: ["wheel", "bolts"],
    remove: ["bolts", "wheel"],
    ref: "",
    when: { caliper: "floating" },
    say: "Remove the wheel bolts and the wheel. If the wheel sticks to the hub, push the tyre sideways by hand: never hit the wheel or the disc with a hammer.",
    tasks: [{ type: "check", items: ["Wheel bolts out and kept clean", "Wheel off, laid flat away from the work area", "Locking wheel bolt adapter kept with the car, if fitted"] }],
  },
  {
    id: "sensor-off",
    group: "remove",
    title: "Unplug the pad wear wire",
    figure: "corner",
    focus: ["innerPad"],
    ref: "",
    when: { caliper: "floating", sensor: "yes" },
    say: "Unclip the wear wire from its holder and separate the connector. The wire wears through with the pad: if the warning light was on, fit a new wire with the new pads.",
    tasks: [{ type: "check", items: ["Connector separated, wire unclipped", "Wire checked for wear: new wire if its tip is worn through"] }],
  },
  {
    id: "caliper-off",
    group: "remove",
    title: "Take the caliper off and hang it up",
    figure: "corner",
    focus: ["caliper"],
    remove: ["caliper"],
    ref: "",
    when: { caliper: "floating" },
    say: "Pry off the caps over the guide pin bolts, unscrew the two guide bolts and lift the caliper off the pads. Hang it from the suspension with a wire or hook: its weight must never hang on the brake hose, and the hose must not twist.",
    tasks: [
      { type: "check", items: ["Pad retaining spring removed from the caliper, if fitted", "Guide pin bolts out", "Caliper hung up, hose slack and not twisted", "Do not press the brake pedal while the caliper is off"] },
      {
        type: "choice",
        q: "The caliper is off. What happens if someone presses the brake pedal now?",
        options: [
          { text: "Nothing: the brake is disconnected", ok: false, why: "The hydraulic line is still connected; only the pads are gone." },
          { text: "The piston is pushed out of the caliper, maybe all the way", ok: true, why: "With no pads to stop it, the piston travels out. Brake fluid can leak and the piston may have to be refitted with special tools." },
          { text: "The parking brake applies", ok: false, why: "The pedal works the hydraulic brakes, not the parking brake." },
        ],
      },
    ],
  },
  {
    id: "pads-off",
    group: "remove",
    title: "Take out the pads",
    figure: "corner",
    focus: ["innerPad", "outerPad"],
    remove: ["innerPad", "outerPad"],
    ref: "",
    when: { caliper: "floating" },
    say: "Take the pads out of the carrier. Note which pad was inner and which outer and how worn each is: uneven wear points to a sticking guide pin or a seized piston.",
    tasks: [{ type: "check", items: ["Pads out, kept in order inner and outer", "Pad wear compared side to side and inner to outer"] }],
  },
  {
    id: "carrier-off",
    group: "remove",
    title: "Take the carrier off",
    figure: "corner",
    focus: ["carrier"],
    remove: ["carrier"],
    ref: "",
    when: { caliper: "floating" },
    say: "Unscrew the two bolts holding the carrier to the wheel bearing housing and take it off. These bolts are tight. Check in your source whether they must be replaced once removed.",
    tasks: [{ type: "check", items: ["Carrier bolts out", "Carrier off", "Bolt replacement rule checked in your source"] }],
  },
  {
    id: "disc-off",
    group: "remove",
    title: "Take the disc off",
    figure: "corner",
    focus: ["disc"],
    remove: ["disc"],
    ref: "",
    when: { caliper: "floating" },
    say: "Remove the small retaining screw and pull the disc off the hub. If it is rusted on, soak the centre with penetrating oil and tap the hat (not the friction ring) with a soft mallet.",
    tasks: [{ type: "check", items: ["Retaining screw out (new one ready)", "Disc off the hub"] }],
  },
  {
    id: "inspect-hub",
    group: "inspect",
    title: "Clean and inspect the hub, carrier and caliper",
    figure: "corner",
    focus: ["hub"],
    ref: "",
    when: { caliper: "floating" },
    say: "Rust on the hub face makes a new disc run out of true, which you will feel as a vibration under braking a few weeks later. Clean the hub face to bare metal and look over the hydraulic parts.",
    tasks: [
      { type: "check", items: ["Hub face cleaned of rust to bare metal, no grease on it", "Carrier pad seats cleaned of rust and old paste", "Guide pins slide freely and their boots are not torn", "Piston dust boot not torn, no fluid leak at the caliper", "Brake hose not cracked, bulged or chafed"] },
      {
        type: "choice",
        q: "One pad was much more worn than the other. What is the likely cause?",
        options: [
          { text: "Normal: the inner pad always wears out first", ok: false, why: "A little difference is normal; much more on one pad is not." },
          { text: "A guide pin or the piston is sticking", ok: true, why: "If the caliper can't slide or the piston doesn't retract, one pad stays in contact. Fix it now, or the new pads will wear the same way." },
          { text: "The disc was too thin", ok: false, why: "A thin disc wears both pads, not just one." },
        ],
      },
    ],
  },
  {
    id: "measure",
    group: "inspect",
    title: "Measure the discs and pads",
    figure: "bench",
    focus: ["disc", "pads"],
    ref: "",
    when: { caliper: "floating" },
    say: "If you are reusing the discs, they must be above the minimum thickness. Measure with a micrometer at several points around the friction ring, about 10 mm in from the edge (the rim lip is unworn and reads high). Measuring the old parts also tells you how they wore.",
    tasks: [
      { type: "measure", spec: "discThickness", label: "Disc thickness, old disc", items: ["Point 1", "Point 2", "Point 3", "Point 4"] },
      { type: "measure", spec: "padThickness", label: "Pad friction material, old pads (without backing plate)", items: ["Inner pad", "Outer pad"] },
      { type: "check", items: ["Discs and pads are always changed in pairs on both sides of the axle"] },
    ],
  },
  {
    id: "disc-on",
    group: "install",
    title: "Fit the new disc",
    figure: "corner",
    focus: ["disc"],
    install: ["disc"],
    ref: "",
    when: { caliper: "floating" },
    say: "New discs are coated against rust; most coatings stay on, so check the box before you clean anything off. Fit the disc flat on the clean hub and secure it with the new retaining screw.",
    tasks: [{ type: "torque", what: "Disc retaining screw (new)", stages: [pending("Retaining screw")] }],
  },
  {
    id: "carrier-on",
    group: "install",
    title: "Fit the carrier",
    figure: "corner",
    focus: ["carrier"],
    install: ["carrier"],
    ref: "",
    when: { caliper: "floating" },
    say: "Fit the carrier to the wheel bearing housing. Use new bolts if your source says so; if they come with thread-locking compound on them, do not add more.",
    tasks: [{ type: "torque", what: "Carrier to wheel bearing housing", tool: "torque wrench", note: "Some carrier bolts take a torque then a further angle: follow your source.", stages: [pending("Carrier bolts"), { bolts: "Further angle, only if your source gives one", check: true }] }],
  },
  {
    id: "piston-back",
    group: "install",
    title: "Push the caliper piston back",
    figure: "corner",
    focus: ["caliper"],
    ref: "",
    when: { caliper: "floating" },
    say: "New pads are thicker than old ones, so the piston has to go back into the caliper. Pushing it back pushes brake fluid up to the reservoir.",
    tasks: [
      { type: "check", items: ["Brake fluid reservoir checked: draw some off if it is at MAX, so it cannot overflow", "Piston pressed back slowly and squarely with a piston resetting tool", "Piston dust boot still seated after pressing back"] },
      {
        type: "choice",
        q: "The reservoir was topped up to MAX when the old pads were worn. What happens as you push the piston back?",
        options: [
          { text: "The fluid level drops", ok: false, why: "Pushing the piston in moves fluid back toward the reservoir, so the level rises." },
          { text: "The reservoir can overflow, and brake fluid damages paint", ok: true, why: "Draw some fluid off with a clean syringe first, and cover painted parts nearby." },
          { text: "Nothing, the fluid goes to the other wheel", ok: false, why: "The fluid goes back up the line to the reservoir." },
        ],
      },
    ],
  },
  {
    id: "pads-on",
    group: "install",
    title: "Fit the new pads",
    figure: "corner",
    focus: ["innerPad", "outerPad"],
    install: ["innerPad", "outerPad"],
    ref: "",
    when: { caliper: "floating" },
    say: "Fit the pads to the carrier. Use only the paste your source names, only where the pad's backing plate touches the carrier, and never on the friction material or the disc.",
    tasks: [{ type: "check", items: ["Pad contact points on the carrier pasted lightly, if your source says to", "Inner pad (with wear wire, if fitted) on the piston side, outer pad on the outside", "Friction material and disc free of paste, grease and fingerprints"] }],
  },
  {
    id: "caliper-on",
    group: "install",
    title: "Fit the caliper",
    figure: "corner",
    focus: ["caliper"],
    install: ["caliper"],
    ref: "",
    when: { caliper: "floating" },
    say: "Lower the caliper over the pads, fit the guide pin bolts and torque them. Then fit the pad retaining spring and the bolt caps, and plug in the wear wire.",
    tasks: [
      { type: "torque", what: "Caliper guide pin bolts", stages: [pending("Guide pin bolts")] },
      { type: "check", items: ["Pad retaining spring fitted, if your caliper has one", "Guide bolt caps fitted", "Wear wire plugged in and clipped clear of the disc"], },
    ],
  },
  {
    id: "wheel-on",
    group: "finish",
    title: "Fit the wheel and tighten the bolts in a star pattern",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    when: { caliper: "floating" },
    say: "Before the wheel goes on, pump the brake pedal until it is firm: the first presses only move the piston up to the new pads. Fit the wheel, screw the bolts in by hand, lower the car and tighten in a star pattern so the wheel pulls on evenly.",
    tasks: [
      { type: "check", items: ["Brake pedal pumped until firm", "Hub spigot and wheel centre clean; no grease on the wheel bolt threads or seats unless your source says so", "Bolts screwed in by hand, car lowered to the ground"] },
      { type: "order", diagram: "wheel", q: "Tighten the wheel bolts in a star pattern: click them in order, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "torque wrench, not an impact gun", stages: [pending("All five, in the star pattern")] },
    ],
  },
  {
    id: "test",
    group: "finish",
    title: "Check and bed in the new brakes",
    figure: "corner",
    focus: ["disc", "innerPad", "outerPad"],
    ref: "",
    when: { caliper: "floating" },
    say: "New pads and discs need bedding in before they brake at full strength. Do the other side of the axle the same way, then road test carefully.",
    tasks: [
      { type: "check", items: ["Other side of the axle done", "Brake fluid level between MIN and MAX", "Pedal firm with the engine running", "Several gentle stops from moderate speed in a safe place, without coming to a full stop and holding the brake", "Hard braking avoided for the first few hundred kilometres, as the pad maker advises", "Wheel bolts re-checked with the torque wrench after a short drive"] },
    ],
  },
];

// Star order for a five-bolt wheel: each bolt is followed by the one across from it.
const star = [0, 2, 4, 1, 3];
const DIAGRAMS = {
  wheel: {
    ref: "",
    shape: "wheel",
    hint: "Go across the hub each time, not round it.",
    bolts: star.map((pos, i) => {
      const t = (-90 + pos * 72) * (Math.PI / 180);
      return { n: i + 1, x: 50 + 30 * Math.cos(t), y: 53 + 30 * Math.sin(t) };
    }),
  },
};
// The wheel pattern is general workshop practice, not from a licensed source.
DIAGRAMS.wheel.ref = "general workshop practice";

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: SPEC, diagrams: DIAGRAMS, figures: buildFigures() };
}
