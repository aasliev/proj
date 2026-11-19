// EA888 Gen 2 (CAEB) rebuild trainer: stage 3, cylinder head, valvetrain and camshafts.
// Source: Audi repair manual EA888 Gen II, Edition 06.2019, Rep. Gr. 15 sections 3–5 (2.0L
// procedures), cited as "repair group · printed page"; paraphrased, never copied.

export const VARIANTS3 = {
  headBolt: {
    q: "Cylinder head bolts",
    help: "Two bolt types exist: with a collar under the head, or without. The first torque step differs (15 · p. 107).",
    options: [
      { id: "plain", label: "Without collar" },
      { id: "collar", label: "With collar" },
    ],
  },
};

export const SPEC3 = {
  headFlat: { min: 0, max: 0.05, unit: "mm", ref: "15 · p. 107" },
  guideTilt: { min: 0, max: 0.6, unit: "mm", ref: "15 · p. 142" },
  stemIn: { nominal: 5.98, dev: 0.01, unit: "mm", ref: "15 · p. 142" },
  stemEx: { nominal: 5.96, dev: 0.01, unit: "mm", ref: "15 · p. 142" },
  camRadial: { min: 0.024, max: 0.066, unit: "mm", ref: "15 · p. 118–119" },
  camRunout: { min: 0, max: 0.04, unit: "mm", ref: "15 · p. 118–119" },
};

const VALVES = (side) => [1, 2, 3, 4].flatMap((c) => [`Cyl ${c} ${side} A`, `Cyl ${c} ${side} B`]);

export const HEAD_STEPS = [
  // ---------------------------------------------------------------- the head on the bench
  {
    id: "head-parts",
    group: "head",
    title: "Head versions and new parts",
    figure: "head",
    ref: "15 · p. 104–119, 134",
    say: "The head, its gasket and the camshaft housing. On this engine the cylinder head cover is the camshaft housing: the camshaft bearings are built into it, so head and cover are a matched pair.",
    tasks: [
      { type: "variant", keys: ["headBolt"] },
      {
        type: "check",
        items: [
          "Cylinder head gasket (unpack only just before fitting) · 15 · p. 112",
          "10 cylinder head bolts and the 4 bolts marked with arrows in the figure (replace after removing) · 15 · p. 104, 107",
          "6 cylinder head cover bolts (replace after removing) · 15 · p. 119",
          "Cap in the cylinder head cover (always replaced once the cover has been off) · 15 · p. 119, 126",
          "Valve stem seals, and a mounting sleeve from the parts catalogue · 15 · p. 134–137",
          "Check valve for the oil channel in the head, if you removed it · 15 · p. 113–114",
          "O-rings and seals on the head (replace after removing) · 15 · p. 105–106, 112",
          "Silicone sealant for the cylinder head cover, in date · 15 · p. 126, 130",
          "New coolant and engine oil: both are changed whenever the head or gasket is replaced · 15 · p. 104, 112",
        ],
      },
      {
        type: "choice",
        q: "The cylinder head cover is damaged. Can you replace just the cover?",
        options: [
          { text: "No: head and cover are only replaced together", ok: true, why: "The camshaft bearings are split between them, so they are machined as a pair." },
          { text: "Yes, covers are interchangeable", ok: false, why: "Only together with the cylinder head." },
        ],
      },
    ],
  },
  {
    id: "head-flat",
    group: "head",
    title: "Check the head for distortion",
    figure: "deck",
    ref: "15 · p. 107",
    say: "Lay a straight edge across the head's gasket face in several places and try feeler gauges under it. Clean the face first, without leaving grooves or scratches.",
    tasks: [
      { type: "measure", spec: "headFlat", label: "Distortion under the straight edge", items: ["Lengthwise, exhaust side", "Lengthwise, middle", "Lengthwise, intake side", "Diagonal 1", "Diagonal 2", "Across, chain end", "Across, middle", "Across, transmission end"] },
      { type: "check", items: ["The upper sealing face of the head (to the cover) is never reworked · 15 · p. 104"] },
    ],
  },
  {
    id: "valves-check",
    group: "head",
    title: "Check the valves",
    figure: "valve",
    focus: ["valve"],
    ref: "15 · p. 142",
    say: "Look at each valve's stem and seat for wear grooves; a valve with clear grooves is replaced. Valves are never reworked: only lapping is allowed.",
    tasks: [
      { type: "measure", spec: "stemIn", label: "Intake valve stem diameter", items: VALVES("intake") },
      { type: "measure", spec: "stemEx", label: "Exhaust valve stem diameter", items: VALVES("exhaust") },
      { type: "note", text: "Other dimensions for reference (15 · p. 142): head diameter intake 33.85 ± 0.10 mm, exhaust 28.0 ± 0.1 mm; length intake 104.0 ± 0.2 mm, exhaust 101.9 ± 0.2 mm; seat angle 45°." },
    ],
  },
  {
    id: "guides",
    group: "head",
    title: "Check the valve guides",
    figure: "valve",
    focus: ["guide"],
    ref: "15 · p. 142",
    say: "Put the valve in its guide with the stem end flush with the guide, and measure how far the valve head tilts with a dial gauge.",
    tasks: [
      {
        type: "choice",
        q: "Can you use an exhaust valve to check an intake guide?",
        options: [
          { text: "No: intake valves in intake guides, exhaust in exhaust", ok: true, why: "The stem diameters differ." },
          { text: "Yes, any valve will do", ok: false, why: "Intake and exhaust stems differ in diameter." },
        ],
      },
      { type: "measure", spec: "guideTilt", label: "Valve tilt in its guide", items: [...VALVES("intake"), ...VALVES("exhaust")] },
      { type: "note", text: "Over 0.60 mm: measure again with a new valve. Still over: the cylinder head is replaced." },
    ],
  },
  {
    id: "stem-seals",
    group: "head",
    title: "Fit the valve stem seals and springs",
    figure: "valve",
    focus: ["seal", "spring", "retainer"],
    install: ["seal", "spring", "retainer"],
    ref: "15 · p. 137–141",
    say: "With the head off the engine, the manual mounts it in tensioning device VAS6419 and uses valve keeper kit VAS5161A. A sleeve over the valve stem protects the new seal on its way down.",
    tasks: [
      {
        type: "choice",
        q: "What goes over the valve stem before you press the new seal on?",
        options: [
          { text: "The mounting sleeve, so the stem's grooves can't cut the seal lip", ok: true, why: "Then remove the sleeve once the seal is fully home." },
          { text: "Nothing: press it straight on", ok: false, why: "The keeper grooves on the stem would damage the lip." },
        ],
      },
      { type: "check", items: ["Seal lip lightly oiled", "Pressed fully onto the valve guide with installer 3365 or T40376/1, depending on the seal's make", "Valve spring and spring retainer fitted, keepers set with VAS5161A", "Repeated on all 16 valves"] },
    ],
  },
  {
    id: "check-valve",
    group: "head",
    title: "Check valve in the oil channel",
    figure: "head",
    ref: "15 · p. 113–114",
    say: "Only if you removed it. The check valve sits in an oil channel of the head.",
    tasks: [{ type: "check", items: ["Oil channel cleaned", "Seal on the new check valve coated with clean engine oil", "Pushed in by hand all the way"] }],
  },
  {
    id: "rockers",
    group: "head",
    title: "Fit the roller rocker levers and lifters",
    figure: "head",
    focus: ["rockers"],
    install: ["rockers"],
    ref: "15 · p. 118, 129, 134",
    say: "Each roller rocker lever and its hydraulic lifter go back where they came from.",
    tasks: [
      {
        type: "choice",
        q: "You lost track of which lifter came from which valve. What does the manual say?",
        options: [
          { text: "They must not be interchanged: mark them at teardown so each goes back to its own place", ok: true, why: "The manual says do not interchange, and to mark the allocation before removing them." },
          { text: "It doesn't matter, they are all the same part", ok: false, why: "Used lifters and levers have bedded in to their own valve and cam." },
        ],
      },
      { type: "check", items: ["Contact surfaces lubricated", "Every roller rocker lever rests on its valve stem end"] },
    ],
  },
  // ---------------------------------------------------------------- the head on the block
  {
    id: "head-gasket",
    group: "head",
    title: "Lay the cylinder head gasket",
    figure: "head",
    focus: ["gasket"],
    install: ["gasket"],
    ref: "15 · p. 104, 112–113",
    say: "Clean the head and block faces carefully, without grooves or scratches, and remove all lapping and sanding residue. Clear oil and coolant out of the head bolt holes in the block.",
    tasks: [
      {
        type: "choice",
        q: "Which way up does the gasket go?",
        options: [
          { text: "Part number toward the head, readable from the intake side", ok: true, why: "That is the installed position. Seat it on the centering pins in the block." },
          { text: "Part number down, toward the block", ok: false, why: "The part number faces the cylinder head." },
        ],
      },
      { type: "check", items: ["Gasket unpacked only now, handled very carefully (its silicone layer is easily damaged)", "Seated on the centering pins", "Bolt holes in the block dry and clean"] },
    ],
  },
  {
    id: "head-on",
    group: "head",
    title: "Set the head on the block",
    figure: "head",
    focus: ["headBody"],
    install: ["headBody", "head"],
    ref: "15 · p. 113, 122",
    say: "With the crank at TDC, pistons 1 and 4 are at the top of their bores. The head goes on with the crank turned back slightly from TDC.",
    tasks: [
      {
        type: "choice",
        q: "Where should the crank be when the head goes on?",
        options: [
          { text: "Piston 1 brought to TDC, then the crank turned back slightly", ok: true, why: "So no piston sits right at the top where an open valve could touch it." },
          { text: "Exactly at TDC", ok: false, why: "The manual says to turn it back slightly from TDC." },
        ],
      },
      { type: "check", items: ["If the head is new: its plastic base stays on until just before it goes on, to protect the open valves"] },
    ],
  },
  {
    id: "head-bolts",
    group: "head",
    title: "Tighten the cylinder head bolts",
    figure: "head",
    focus: ["headBolts"],
    install: ["headBolts"],
    ref: "15 · p. 107, 113",
    say: "Six steps: the ten main bolts in a spiral from the middle, then the four bolts the manual marks with arrows. No retightening is needed later.",
    tasks: [
      { type: "order", diagram: "head10", q: "Click the ten head bolts in tightening order." },
      {
        type: "torque",
        what: "Cylinder head bolts without collar (new)",
        when: { headBolt: "plain" },
        stages: [{ bolts: "Bolts 1–10: in by hand all the way", check: true }, { bolts: "Bolts 1–10", nm: 40 }, { bolts: "Bolts 1–10", deg: 90 }, { bolts: "Bolts 1–10", deg: 90 }, { bolts: "Four arrowed bolts (manual figure, p. 107)", nm: 8 }, { bolts: "Four arrowed bolts (manual figure, p. 107)", deg: 90 }],
      },
      {
        type: "torque",
        what: "Cylinder head bolts with collar (new)",
        when: { headBolt: "collar" },
        stages: [{ bolts: "Bolts 1–10: in by hand all the way", check: true }, { bolts: "Bolts 1–10", nm: 50 }, { bolts: "Bolts 1–10", deg: 90 }, { bolts: "Bolts 1–10", deg: 90 }, { bolts: "Four arrowed bolts (manual figure, p. 107)", nm: 8 }, { bolts: "Four arrowed bolts (manual figure, p. 107)", deg: 90 }],
      },
    ],
  },
  // ---------------------------------------------------------------- camshafts and their housing
  {
    id: "cam-clearance",
    group: "head",
    title: "Check camshaft clearance and run-out",
    figure: "head",
    focus: ["cams"],
    ref: "15 · p. 118–119",
    say: "Radial clearance with Plastigauge, with the roller rocker levers removed so the camshaft sits on its bearings; run-out with a dial gauge.",
    tasks: [
      { type: "measure", spec: "camRadial", label: "Camshaft radial clearance", items: ["Intake camshaft", "Exhaust camshaft"] },
      { type: "measure", spec: "camRunout", label: "Camshaft run-out", items: ["Intake camshaft", "Exhaust camshaft"] },
    ],
  },
  {
    id: "sliding-ball",
    group: "head",
    title: "Sliding bar ball and spring",
    figure: "head",
    focus: ["cams"],
    ref: "15 · p. 132",
    say: "The valvelift system's sliding bar is held by a small spring-loaded ball in the camshaft. Wear eyewear: the ball can spring out.",
    tasks: [{ type: "check", items: ["Protective eyewear on", "Spring into the camshaft, ball on the spring", "Ball and spring held down with T40191/1 while the sliding bar is pushed over"] }],
  },
  {
    id: "cams-in",
    group: "head",
    title: "Lay in the camshafts",
    figure: "head",
    focus: ["cams"],
    install: ["cams"],
    ref: "15 · p. 129–130",
    say: "The crank must not be at TDC. Lubricate the running surfaces and lay both camshafts in with camshaft spacers T40191, factory markings positioned as the manual shows.",
    tasks: [
      {
        type: "choice",
        q: "You marked the camshafts yourself at teardown. Which marks set them in the head?",
        options: [
          { text: "The factory markings; ignore your own here", ok: true, why: "Your marks are for the camshaft locks later. Copy them to new camshafts if you replaced them." },
          { text: "Your own marks", ok: false, why: "Use the factory markings here; your marks help with the camshaft locks later." },
        ],
      },
      { type: "check", items: ["Crank at TDC then turned back slightly, not at TDC", "Running surfaces lubricated", "Alignment of both camshafts checked"] },
    ],
  },
  {
    id: "cam-housing",
    group: "head",
    title: "Fit the cylinder head cover (camshaft housing)",
    figure: "head",
    focus: ["housing"],
    install: ["housing"],
    ref: "15 · p. 118, 129–130",
    say: "Clean the old sealant out of the cover's groove and off the sealing faces with a rotating plastic brush, eyewear on, then degrease.",
    tasks: [
      {
        type: "choice",
        q: "Bead and timing for the silicone sealant?",
        options: [
          { text: "Nozzle about 2 mm, bead 2 mm, cover on within five minutes, not tilted", ok: true, why: "Thicker sealant can get into the oil pan and block the pickup screen." },
          { text: "A generous bead so it seals", ok: false, why: "2 mm and no more." },
        ],
      },
      { type: "order", diagram: "cover6", q: "Click the six cover bolts in order." },
      { type: "torque", what: "Cylinder head cover bolts (new)", stages: [{ bolts: "Bolts 1–6: in by hand all the way, in several steps", check: true }, { bolts: "Bolts 1–6", nm: 8 }, { bolts: "Bolts 1–6", deg: 90 }] },
      { type: "check", items: ["New cap driven in without sealant with T10174, to a depth of 1–2 mm · 15 · p. 130"] },
    ],
  },
  {
    id: "cam-lock",
    group: "head",
    title: "Lock the camshafts at TDC",
    figure: "head",
    focus: ["cams"],
    ref: "15 · p. 130–131, 13 · p. 25",
    say: "Turn the crank to TDC, in the direction of engine rotation only. The manual uses the damper notch for this; in a full rebuild the damper isn't on yet, so use the turning-over tool T10531/3: its flat side is at the top at TDC.",
    tasks: [
      { type: "check", items: ["Crank at TDC: turning-over tool flat side up", "Intake camshaft turned until its marking lines up with T40271/2; lock slid into the sprocket splines", "Exhaust camshaft turned until its marking lines up with T40271/1; lock slid into the sprocket splines"] },
      {
        type: "choice",
        q: "Next is the camshaft timing chain. Which way may the engine turn from now on?",
        options: [
          { text: "Only in its direction of rotation", ok: true, why: "Turning it backward can let the timing skip." },
          { text: "Either way, gently", ok: false, why: "Only in the direction of engine rotation." },
        ],
      },
    ],
  },
];

// After the camshaft timing chain is on.
export const AFTER_CHAIN_STEPS = [
  {
    id: "avs-actuators",
    group: "timing",
    title: "Fit the valvelift actuators",
    figure: "head",
    focus: ["actuators"],
    install: ["actuators"],
    ref: "15 · p. 118–119, 133",
    say: "The cam adjustment actuators of the Audi valvelift system go on after the camshaft timing chain.",
    tasks: [
      {
        type: "choice",
        q: "Before fitting an actuator, where must its pin be?",
        options: [
          { text: "Pressed down by hand into the installation position, not out", ok: true, why: "An extended pin would hit the camshaft's sliding pieces." },
          { text: "Out, so it engages straight away", ok: false, why: "Press it down by hand first." },
        ],
      },
      { type: "check", items: ["O-rings checked for damage and oiled (they come with the actuator)"] },
      { type: "torque", what: "Cam adjustment actuator bolts", stages: [{ bolts: "Each bolt", nm: 5 }] },
    ],
  },
];

export const DIAGRAMS3 = {
  // As in the manual's figure on p. 107.
  head10: {
    ref: "15 · p. 107",
    shape: "pan",
    bolts: [
      { n: 10, x: 10, y: 26 }, { n: 3, x: 31, y: 26 }, { n: 1, x: 52, y: 26 }, { n: 5, x: 73, y: 26 }, { n: 7, x: 94, y: 26 },
      { n: 8, x: 10, y: 74 }, { n: 6, x: 31, y: 74 }, { n: 2, x: 52, y: 74 }, { n: 4, x: 73, y: 74 }, { n: 9, x: 94, y: 74 },
    ],
  },
  cover6: {
    ref: "15 · p. 118",
    shape: "pan",
    bolts: [{ n: 6, x: 8, y: 50 }, { n: 4, x: 24, y: 50 }, { n: 2, x: 42, y: 50 }, { n: 1, x: 60, y: 50 }, { n: 3, x: 77, y: 50 }, { n: 5, x: 93, y: 50 }],
  },
};
