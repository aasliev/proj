// EA888 Gen 2 (CAEB): coolant leak at the coolant pump and thermostat housing. Pressure test,
// pump and thermostat replacement and thermostat test, from the Audi repair manual, Rep. Gr. 19,
// p. 174–183. PRIVATE.
import { manualMeta } from "../../lib/ea888.mjs";
import { buildFigures } from "../../../docs/figures/ea888-caeb/rebuild/figures.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";

export const meta = manualMeta({
  id: "ea888-coolant-pump-leak",
  title: "Coolant leak: pump and thermostat housing",
  system: "Cooling",
  commonProblem: "Coolant pump and thermostat housing leaks",
  minutes: 300,
  difficulty: "Intermediate",
  summary: "Find a coolant leak with a pressure test, then replace the coolant pump and thermostat, which share a plastic housing that commonly leaks on this engine.",
});

const SPEC = {
  // Pressure drop allowed in 10 minutes at about 1 bar (14.5 psi) test pressure.
  pressureDrop: { min: 0, max: 2.9, unit: "psi", ref: "19 · p. 175" },
  thermoLift: { min: 8, unit: "mm", ref: "19 · p. 183" },
};

const GROUPS = [
  { id: "find", name: "Find the leak" },
  { id: "remove", name: "Remove" },
  { id: "install", name: "Install" },
  { id: "fill", name: "Fill and check" },
];

const STEPS = [
  {
    id: "symptoms",
    group: "find",
    title: "Signs of a pump or thermostat housing leak",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "",
    say: "Commonly reported on this engine: the coolant level drops slowly, there's a sweet smell after driving, and pink or purple crust or drips appear below the intake side of the engine, under the intake manifold where the coolant pump and thermostat housing sit. Overheating or a temperature gauge that swings can follow.",
    tasks: [
      { type: "check", items: ["Coolant level history noted (how much and how often it was topped up)", "Underside and intake side looked at with a flashlight: crust or drips below the intake manifold", "Oil checked for milky deposits under the filler cap (coolant in the oil is a different, more serious problem)"] },
    ],
  },
  {
    id: "pressure-test",
    group: "find",
    title: "Pressure-test the cooling system",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "19 · p. 174–175",
    say: "With the engine at operating temperature, release the pressure slowly: cover the expansion tank cap with a cloth and open it carefully. Fit the cooling system tester to the tank and pump it to about 1 bar (14.5 psi). Over 10 minutes the pressure must not drop more than 0.2 bar (2.9 psi).",
    tasks: [
      { type: "check", items: ["Safety gloves and eye protection on: hot coolant under pressure can scald · 19 · p. 174", "Cap opened slowly under a cloth", "Tester VAG1274B with adapter VAG1274/8 on the expansion tank", "Pumped to about 1 bar (14.5 psi)"] },
      { type: "measure", spec: "pressureDrop", label: "Pressure drop after 10 minutes", items: ["Drop"] },
      { type: "note", text: "A small drop is normal as the coolant cools; the colder the engine, the bigger the drop. If in doubt, repeat the test with the engine cold (19 · p. 175)." },
      { type: "check", items: ["If it dropped more than 2.9 psi: leak found while under pressure (look at the pump and thermostat housing, hoses, radiator and heater connections)", "Pressure released with the tester's release valve until the gauge reads 0"] },
    ],
  },
  {
    id: "drain",
    group: "remove",
    title: "Drain the coolant and reach the pump",
    figure: "coolant",
    focus: ["guard"],
    ref: "19 · p. 177, 180",
    say: "The pump sits on the intake side of the block, driven by a toothed belt from the balance shaft. Drain the coolant and do the model-specific preliminary work the vehicle manual describes (the engine manual refers to Rep. Gr. 19 of the vehicle manual for both). Used coolant can't be reused.",
    tasks: [{ type: "check", items: ["Coolant drained into a clean container for disposal (used coolant cannot be reused) · 19 · p. 177", "Preliminary work for your model done, from the vehicle manual's Rep. Gr. 19", "New coolant pump (with thermostat if replacing it too), seals and O-rings, and a new drive wheel bolt · 19 · p. 178–179"] }],
  },
  {
    id: "belt-off",
    group: "remove",
    title: "Remove the belt guard and the pump's toothed belt",
    figure: "coolant",
    focus: ["guard", "belt", "wheel"],
    remove: ["guard", "belt"],
    ref: "19 · p. 181",
    say: "Remove the guard bolts and the guard. The drive wheel bolt has a left-hand thread: loosen it clockwise and unscrew it three turns, then take the belt off.",
    tasks: [
      { type: "check", items: ["Toothed belt guard off", "Drive wheel bolt loosened three turns with VAG1410 and T10360 · 19 · p. 181", "Toothed belt off"] },
      {
        type: "choice",
        q: "Which way do you turn the drive wheel bolt to loosen it?",
        options: [
          { text: "Counterclockwise, like any bolt", ok: false, why: "It has a left-hand thread. Turning it counterclockwise tightens it." },
          { text: "Clockwise: it has a left-hand thread", ok: true, why: "The manual notes the left-hand thread (19 · p. 179, 181)." },
        ],
      },
    ],
  },
  {
    id: "pump-off",
    group: "remove",
    title: "Remove the coolant pump",
    figure: "coolant",
    focus: ["pump"],
    remove: ["pump"],
    ref: "19 · p. 180",
    say: "Unplug the coolant temperature sensor G62, open the hose clamps and move the hoses aside. Pull the pump off its centering pins and away from the engine oil cooler connection.",
    tasks: [{ type: "check", items: ["G62 connector unplugged", "Hoses off and moved aside", "Pump off the centering pins and the oil cooler connection piece", "Sealing faces on the block cleaned"] }],
  },
  {
    id: "thermostat",
    group: "remove",
    title: "Thermostat: replace or test",
    figure: "coolant",
    focus: ["pump"],
    ref: "19 · p. 182–183",
    say: "The thermostat sits in the pump housing, behind a connection held by bolts. If you are keeping it, test it in a hot water bath: it starts opening at about 95 °C (203 °F), is fully open at about 105 °C (221 °F) and must lift at least 8 mm.",
    tasks: [
      { type: "measure", spec: "thermoLift", label: "Thermostat opening lift (only if testing the old one)", items: ["Lift"] },
      { type: "check", items: ["Thermostat installed into the pump housing and moved slightly forward, as the manual's figure shows · 19 · p. 182", "Connection fitted with its centering pin in the guide", "New seals and O-ring, O-ring coated with coolant · 19 · p. 182"] },
    ],
  },
  {
    id: "pump-on",
    group: "install",
    title: "Install the coolant pump",
    figure: "coolant",
    focus: ["pump"],
    install: ["pump"],
    ref: "19 · p. 178–180",
    say: "New seals and O-rings, O-rings coated with coolant. Make sure both centering pins are in the block, fit the connection piece into the oil cooler, then push the pump onto it and the pins. On a new pump, remove its protective cap.",
    tasks: [
      { type: "check", items: ["New seal and O-rings, coated with coolant", "Both centering pins in the block", "Connection piece in the oil cooler, pump pushed onto it and the pins", "Protective cap removed from the new pump · 19 · p. 180"] },
      { type: "torque", what: "Coolant pump bolts", stages: [{ bolts: "Bolts 1–5, in sequence", nm: 9 }] },
    ],
  },
  {
    id: "belt-on",
    group: "install",
    title: "Toothed belt, drive wheel and guard",
    figure: "coolant",
    focus: ["belt", "wheel", "guard"],
    install: ["belt", "guard"],
    ref: "19 · p. 179, 181",
    say: "Fit the toothed belt, with the drive wheel in its installed position. The drive wheel bolt is new (it's tightened with an extra turn) and has a left-hand thread.",
    tasks: [
      { type: "torque", what: "Drive wheel bolt (new, left-hand thread)", stages: [{ bolts: "Drive wheel bolt", nm: 10 }, { bolts: "Drive wheel bolt", deg: 90 }] },
      { type: "torque", what: "Toothed belt guard bolts", stages: [{ bolts: "Guard bolts", nm: 9 }] },
      { type: "check", items: ["Hoses on, clamps closed", "G62 connector plugged in"] },
    ],
  },
  {
    id: "fill",
    group: "fill",
    title: "Fill, bleed and re-test",
    figure: "bay",
    focus: ["coolantTank"],
    ref: "19 · p. 177",
    say: "Fill with the coolant additive listed in the parts catalog, mixed with distilled water, protecting to at least −25 °C (−13 °F). Fill and bleed as the vehicle manual's Rep. Gr. 19 describes. Then pressure-test again.",
    tasks: [
      { type: "check", items: ["Coolant additive from the parts catalog, with distilled water · 19 · p. 177", "Freeze protection at least −13 °F (−25 °C), checked with a refractometer", "Filled and bled as the vehicle manual describes", "Pressure test repeated: drop no more than 2.9 psi in 10 minutes", "Level rechecked cold the next day"] },
      {
        type: "choice",
        q: "It's summer. Can you top up with plain water instead of the mix?",
        options: [
          { text: "Yes, there's no frost in summer", ok: false, why: "The additive also prevents corrosion and raises the boiling point. The manual says never to weaken the mix, even in warm seasons." },
          { text: "No: keep the mix at the right strength all year", ok: true, why: "The manual requires at least −25 °C (−13 °F) protection year-round (19 · p. 177)." },
        ],
      },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: SPEC, diagrams: {}, figures: { coolant: buildFigures().coolant, bay: bay() } };
}
