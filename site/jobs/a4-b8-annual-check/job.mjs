// Audi A4 (B8) annual check: a DRAFT inspection checklist that ties the other service jobs
// together. Intervals and limits come from the service schedule in licensed data.
import { draftMeta } from "../../lib/a4b8.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";
import { buildFigures } from "../a4-b8-front-brakes/figures.mjs";
import { wiper } from "../../lib/small-figures.mjs";

export const meta = draftMeta({
  id: "a4-b8-annual-check",
  title: "Annual check",
  system: "Service",
  minutes: 90,
  difficulty: "Beginner",
  summary: "A once-a-year look over the whole car: fluids, lights, wipers, tyres, brakes and underneath, with a list of what's due from the service schedule.",
  draftNote: "The checks are general workshop practice. No licensed data is loaded yet, so the service schedule's intervals and the wear limits show as “spec not loaded”. Each item that needs work links to the job that covers it, where there is one.",
});

const SPEC = {
  tread: { pending: true, unit: "mm", why: "legal minimum depends on your country (1.6 mm in the EU and UK)" },
  pad: { pending: true, unit: "mm", why: "wear limit from your licensed data" },
};
const CORNERS = ["Front left", "Front right", "Rear left", "Rear right"];

const GROUPS = [
  { id: "plan", name: "Plan" },
  { id: "bay", name: "Under the bonnet" },
  { id: "outside", name: "Outside" },
  { id: "under", name: "Wheels, brakes, underneath" },
  { id: "end", name: "Road test and records" },
];

const STEPS = [
  {
    id: "due",
    group: "plan",
    title: "What's due from the schedule",
    figure: "bay",
    focus: ["engine"],
    ref: "",
    say: "Start with the service record and the schedule in your source: some items are due by time, some by distance, whichever comes first. Note what's due so you can plan the parts.",
    tasks: [{ type: "check", items: ["Service record and the car's service reminder read", "Oil change due? (see the oil change job)", "Brake fluid due by time? (see the brake fluid job)", "Air and cabin filters due? (see their jobs)", "Spark plugs, gearbox oil, coolant or other long-interval items due? (spec not loaded)"] }],
  },
  {
    id: "fluids",
    group: "bay",
    title: "Fluids",
    figure: "bay",
    focus: ["oilCap", "coolantTank", "brakeReservoir", "washerCap"],
    ref: "",
    say: "Check all four fluids as in the fluid check job, engine cold for the coolant.",
    tasks: [{ type: "check", items: ["Engine oil level", "Coolant level, engine cold", "Brake fluid level and colour", "Washer fluid, with antifreeze in winter"] }],
  },
  {
    id: "bay-look",
    group: "bay",
    title: "Belts, hoses, battery",
    figure: "bay",
    focus: ["engine"],
    ref: "",
    say: "With a torch, look over everything you can see. Signs of trouble: cracks, wetness, white or green crusts, and anything rubbing.",
    tasks: [{ type: "check", items: ["Auxiliary belt: no cracks, fraying or glazing", "Coolant hoses: no cracks, swelling or crusty leaks at the joints", "No oil leaks on top of or around the engine", "Battery terminals clean and tight; resting voltage tested if starting is slow (see the battery job)", "No chewed wiring or nests (animals like engine bays)"] }],
  },
  {
    id: "lights",
    group: "outside",
    title: "Lights and wipers",
    figure: "wiper",
    focus: ["blade"],
    ref: "",
    say: "With a helper, or using a reflection in a window or garage door, check every light. Then the wipers and washers.",
    tasks: [{ type: "check", items: ["Dipped and main beam, both sides", "Side lights, number plate lights, fog lights", "Indicators front, side and rear, and hazards", "Brake lights, including the high-level one", "Reversing light", "Wipers clear without streaks (see the wiper job); washers aim onto the glass", "Horn works"] }],
  },
  {
    id: "tyres",
    group: "under",
    title: "Tyres",
    figure: "corner",
    focus: ["wheel"],
    ref: "",
    say: "Measure the tread at the shallowest groove of each tyre, and look at both sidewalls. Uneven wear across the tread points to wrong pressures or wheel alignment.",
    tasks: [
      { type: "measure", spec: "tread", label: "Tread depth, shallowest groove", items: CORNERS },
      { type: "check", items: ["No cuts, bulges or exposed cords", "Wear even across each tyre", "Pressures set cold to the fuel flap label", "Spare wheel or repair kit present and usable"] },
    ],
  },
  {
    id: "brakes",
    group: "under",
    title: "Brakes",
    figure: "corner",
    focus: ["innerPad", "outerPad", "disc"],
    ref: "",
    say: "Look through the wheel spokes, or with the wheels off, at the pads and discs. Measure what you can see of the outer pads' friction material.",
    tasks: [
      { type: "measure", spec: "pad", label: "Outer pad friction material", items: CORNERS },
      { type: "check", items: ["Discs: no deep grooves, cracks or large lip at the edge", "Brake hoses not cracked or chafed", "No fluid leaks at the calipers", "Parking brake holds the car on a slope"] },
    ],
  },
  {
    id: "under",
    group: "under",
    title: "Underneath",
    figure: "corner",
    focus: ["hub"],
    ref: "",
    say: "With the car safely on axle stands or ramps, look underneath with a torch. Torn rubber boots let grit into joints, which then wear quickly.",
    tasks: [{ type: "check", items: ["Car supported safely: never under a car on a jack alone", "Drive shaft (CV joint) boots not torn, no grease thrown around", "Steering rack boots and track rod ends not torn", "Suspension arm bushes not cracked; no fluid on the shock absorbers", "No oil or coolant drips from the engine or gearbox", "Exhaust hangers in place, no blowing or rattling", "Undertray complete and secure"] }],
  },
  {
    id: "road",
    group: "end",
    title: "Road test and records",
    figure: "bay",
    focus: ["engine"],
    ref: "",
    say: "A short drive tells you what a look can't. Then write down what you found and what you did, so next year's check starts from it.",
    tasks: [{ type: "check", items: ["No warning lights on the dashboard", "Car drives straight; steering wheel centred", "Brakes stop straight with no judder or noise", "No knocks over bumps", "Air conditioning blows cold; heater blows hot", "Findings and work done written into the service record, with date and mileage"] }],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: SPEC, diagrams: {}, figures: { bay: bay(), corner: buildFigures().corner, wiper: wiper() } };
}
