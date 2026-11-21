// EA888 Gen 2 (CAEB) rebuild trainer: stage 2, timing drive, covers, oil pans and coolant pump.
// Same sources and rules as steps.mjs: Audi repair manual EA888 Gen II, Edition 06.2019,
// cited as "repair group · printed page"; paraphrased, never copied.

export const VARIANTS2 = {
  camTensioner: {
    q: "Cam chain tensioner version",
    help: "Two tensioners exist. Version 1 has a locking wedge and is held with locking pin T40011; version 2 has a circlip and is held with tensioner locking tool T40267 (15 · p. 79–80).",
    options: [
      { id: "v1", label: "Version 1 (locking wedge)" },
      { id: "v2", label: "Version 2 (circlip)" },
    ],
  },
  bracketBolt: {
    q: "Bearing bracket bolt 7",
    help: "The single angle-tightened bolt in the bearing bracket is M6 or M8 depending on the version; the pretension differs (15 · p. 75).",
    options: [
      { id: "M6", label: "M6" },
      { id: "M8", label: "M8" },
    ],
  },
  lowerCover: {
    q: "Lower timing chain cover",
    help: "Count the cover's bolts: 15 or 8, depending on the version (15 · p. 67–70).",
    options: [
      { id: "15", label: "15 bolts" },
      { id: "8", label: "8 bolts" },
    ],
  },
};

export const SPEC2 = {
  timingIntake: { min: 61, max: 64, unit: "mm", ref: "15 · p. 102" },
  timingSpread: { min: 124, max: 126, unit: "mm", ref: "15 · p. 103" },
};

export const GROUPS2 = {
  before: [
    { id: "front", name: "Front: balance shaft drive" },
    { id: "head", name: "Cylinder head (stage 3)" },
    { id: "timing", name: "Camshaft timing chain" },
    { id: "covers", name: "Timing covers and damper" },
    { id: "pan", name: "Upper oil pan" },
  ],
  after: [
    { id: "lube", name: "Oil pump and lower pan" },
    { id: "coolant", name: "Coolant pump" },
  ],
};

// Steps that go before the rear crank seal (the seal needs the upper oil pan on first).
export const BEFORE_REAR = [
  // ---------------------------------------------------------------- front: balance shaft drive
  {
    id: "front-parts",
    group: "front",
    title: "Versions and new parts for the timing and pans",
    figure: "front",
    ref: "15 · p. 66–77, 17 · p. 151–158, 13 · p. 14–17",
    say: "Stage 2 has three more version choices and its own list of parts that are always replaced.",
    tasks: [
      { type: "variant", keys: ["camTensioner", "bracketBolt", "lowerCover"] },
      {
        type: "check",
        items: [
          "Lower timing chain cover with its damper seal (always replaced: the sealant bends it on removal) · 15 · p. 67, 69",
          "Lower timing chain cover bolts (replace after removing) · 15 · p. 67",
          "Upper timing chain cover seal and O-rings (replace) · 15 · p. 66, 69",
          "Bearing bracket bolt 7 (replace after removing) · 15 · p. 74",
          "Vibration damper bolt with its O-ring (replace after removing) · 13 · p. 14, 27",
          "Upper oil pan bolts and O-rings (replace after removing) · 17 · p. 153, 158",
          "Oil baffle (always replaced: its plastic ribs deform) · 17 · p. 152, 154",
          "Lower oil pan bolts and drain plug (replace after removing) · 17 · p. 153",
          "O-rings for the oil pump and suction line, seal for the oil level sensor · 17 · p. 152",
          "Coolant pump seals and O-rings; coolant pump drive wheel bolt (replace) · 19 · p. 178–179",
          "Sealant for the chain tensioner, lower timing cover and both oil pans, from the parts catalog, in date · 15 · p. 70, 17 · p. 155, 159",
        ],
      },
    ],
  },
  {
    id: "sprocket3",
    group: "front",
    title: "Fit the three-stage chain sprocket",
    figure: "front",
    focus: ["sprocket3"],
    install: ["sprocket3"],
    ref: "15 · p. 74–75, 13 · p. 23–25",
    say: "One sprocket on the crank nose drives all three chains: balance shafts, camshafts and oil pump. The vibration damper bolt is what clamps it to the crank, so until the damper goes on it is held by the damper tool's tensioning pin.",
    tasks: [
      {
        type: "choice",
        q: "How do you know the sprocket is in its installed position?",
        options: [
          { text: "The two marked surfaces on the sprocket and the crank line up", ok: true, why: "The manual's figure shows both surfaces aligned." },
          { text: "Any position: it only fits one way", ok: false, why: "Check the alignment of the two surfaces the manual marks." },
        ],
      },
      { type: "check", items: ["Sprocket secured to the crank with tensioning pin T10531/2 while the damper is off · 13 · p. 24–25"] },
    ],
  },
  {
    id: "bal-chain",
    group: "front",
    title: "Install the balance shaft drive chain",
    figure: "front",
    focus: ["balChain", "balRails"],
    install: ["balChain", "balRails"],
    ref: "15 · p. 76, 100–101",
    say: "Turn the intermediate sprocket and the balance shaft to their marks, then lay the chain on with its painted links on the sprocket marks.",
    tasks: [
      {
        type: "choice",
        q: "Where do the chain's painted links go?",
        options: [
          { text: "On the timing marks of the sprockets", ok: true, why: "Painted links on the sprocket marks, then check them again once the tensioner is in." },
          { text: "Anywhere: the tensioner takes up the slack", ok: false, why: "The painted links set the balance shafts' phase to the crank." },
        ],
      },
      { type: "torque", what: "Guide pins for the two guide rails and the tensioning rail", stages: [{ bolts: "Each guide pin", nm: 20 }] },
    ],
  },
  {
    id: "bal-tensioner",
    group: "front",
    title: "Fit the balance chain tensioner",
    figure: "front",
    focus: ["balTensioner"],
    install: ["balTensioner"],
    ref: "15 · p. 76, 100–101",
    say: "The balance chain tensioner screws into the block.",
    tasks: [
      {
        type: "choice",
        q: "What goes on the tensioner's seal?",
        options: [
          { text: "Sealant from the parts catalog", ok: true, why: "The manual says to coat this seal with sealant." },
          { text: "Engine oil", ok: false, why: "This one gets sealant, not oil." },
        ],
      },
      { type: "torque", what: "Balance shaft chain tensioner", stages: [{ bolts: "Tensioner", nm: 85 }] },
      { type: "check", items: ["Marks on the intermediate sprocket and balance shaft checked again"] },
    ],
  },
  // ---------------------------------------------------------------- cylinder head (stage 3)
  {
    id: "head-gate",
    group: "head",
    title: "Cylinder head and camshafts (stage 3, coming next)",
    figure: "front",
    focus: ["head"],
    install: ["head"],
    ref: "15 · p. 104–141",
    say: "The camshaft timing chain goes on with the head and both camshafts already installed. That work is stage 3 of this trainer and isn't built yet: until it is, follow your manual's cylinder head and valvetrain sections, then come back here.",
    tasks: [
      {
        type: "check",
        items: [
          "Cylinder head, gasket, head bolts, camshaft housing and camshafts installed per the manual · 15 · p. 104–141",
          "Crank at TDC; camshafts held by camshaft locks T40271/1 (exhaust) and T40271/2 (intake) · 15 · p. 85–86",
        ],
      },
    ],
  },
  // ---------------------------------------------------------------- camshaft timing chain
  {
    id: "cam-chain",
    group: "timing",
    title: "Lay on the camshaft timing chain",
    figure: "front",
    focus: ["camChain"],
    install: ["camChain"],
    ref: "15 · p. 86–87, 95",
    say: "Both camshafts are locked and the crank is at TDC. The painted links of the chain go on the sprocket marks, in a set order.",
    tasks: [
      {
        type: "choice",
        q: "In what order does the chain go on?",
        options: [
          { text: "Intake camshaft, then exhaust camshaft, then lay it on the crank and hold it", ok: true, why: "That is the manual's order, each painted link on its mark." },
          { text: "Crank first, then the camshafts", ok: false, why: "Start at the intake camshaft, then the exhaust, then the crank." },
        ],
      },
      {
        type: "choice",
        q: "Before the chain is fully tensioned, can you turn the crank to check something?",
        options: [
          { text: "No: never turn the crank while the cam chain is off or loose", ok: true, why: "The valves can hit the piston crowns." },
          { text: "Yes, a little either way", ok: false, why: "Never turn the crank with the cam chain off; and only ever turn the engine in its direction of rotation." },
        ],
      },
    ],
  },
  {
    id: "cam-rails",
    group: "timing",
    title: "Fit the guide rails and tensioning rail",
    figure: "front",
    focus: ["camRails"],
    install: ["camRails"],
    ref: "15 · p. 74, 87–88, 95–97",
    say: "Fixed guide rail and upper guide rail first. Then the tensioning rail goes on while a helper holds the exhaust camshaft, so the chain stays on its marks.",
    tasks: [
      { type: "torque", what: "Guide pins for the guide rail and the tensioning rail", stages: [{ bolts: "Each guide pin", nm: 20 }] },
      {
        type: "check",
        items: [
          "Upper guide rail clipped in",
          "Exhaust camshaft turned slightly so lock T40271/1 slides out; camshaft held while the tensioning rail goes on (second person needed) · 15 · p. 96",
          "Intake camshaft turned slightly so lock T40271/2 slides out, then released · 15 · p. 97",
          "Painted links still on the sprocket marks",
        ],
      },
    ],
  },
  {
    id: "oil-chain",
    group: "timing",
    title: "Fit the oil pump drive chain and its tensioner",
    figure: "front",
    focus: ["oilChain"],
    install: ["oilChain"],
    ref: "15 · p. 98, 17 · p. 152",
    say: "The oil pump chain runs from the third stage of the crank sprocket down to the pump. The pump itself goes in later, from below, and is guided into this chain.",
    tasks: [
      { type: "torque", what: "Oil pump chain tensioner bolt", stages: [{ bolts: "Bolt", nm: 9 }] },
      { type: "check", items: ["Locking pin T40011 removed from the oil pump chain tensioner after the bolt is tight"] },
    ],
  },
  {
    id: "cam-tensioner",
    group: "timing",
    title: "Fit the camshaft chain tensioner",
    figure: "front",
    focus: ["camTensioner"],
    install: ["camTensioner"],
    ref: "15 · p. 74, 98",
    say: "The tensioner goes in held compressed, and stays locked until the bearing bracket is on. Note: the manual's text here points to a cylinder head page for the torque; its chain-drive overview gives the tensioner bolts as 9 Nm.",
    tasks: [
      { type: "torque", what: "Chain tensioner bolts", stages: [{ bolts: "Both bolts", nm: 9 }] },
      { type: "check", when: { camTensioner: "v1" }, items: ["Tensioner held with locking pin T40011"] },
      { type: "check", when: { camTensioner: "v2" }, items: ["Tensioner held with tensioner locking tool T40267"] },
    ],
  },
  {
    id: "bracket",
    group: "timing",
    title: "Fit the bearing bracket",
    figure: "front",
    focus: ["bracket"],
    install: ["bracket"],
    ref: "15 · p. 75, 98",
    say: "Oil the bracket's oil hole, mount it without tilting, run the bolts in by hand, then release the chain tensioner before tightening.",
    tasks: [
      { type: "check", items: ["Oil hole lubricated with engine oil", "Bracket mounted square, not tilted", "Tensioner lock (T40011 or T40267) removed before tightening"] },
      { type: "order", diagram: "bracket7", q: "Click the bearing bracket bolts in order." },
      {
        type: "torque",
        what: "Bearing bracket, bolt 7 new",
        when: { bracketBolt: "M6" },
        stages: [{ bolts: "Bolts 1–7: screw in by hand all the way", check: true }, { bolts: "Bolts 1–6", nm: 9 }, { bolts: "Bolt 7, pretension", nm: 8 }, { bolts: "Bolt 7", deg: 90 }],
      },
      {
        type: "torque",
        what: "Bearing bracket, bolt 7 new",
        when: { bracketBolt: "M8" },
        stages: [{ bolts: "Bolts 1–7: screw in by hand all the way", check: true }, { bolts: "Bolts 1–6", nm: 9 }, { bolts: "Bolt 7, pretension", nm: 20 }, { bolts: "Bolt 7", deg: 90 }],
      },
    ],
  },
  {
    id: "control-valve",
    group: "timing",
    title: "Fit the control valve",
    figure: "front",
    focus: ["controlValve"],
    install: ["controlValve"],
    ref: "15 · p. 74, 98",
    say: "The control valve is the central bolt of the intake camshaft's adjuster. It goes in with the central valve assembly tool.",
    tasks: [
      {
        type: "choice",
        q: "Which way does the control valve tighten?",
        options: [
          { text: "Anticlockwise: it has a left-hand thread", ok: true, why: "The control valves have a left-hand thread." },
          { text: "Clockwise, like most bolts", ok: false, why: "It is a left-hand thread." },
        ],
      },
      { type: "torque", what: "Control valve", tool: "T10352A or T10352/1A, by version", stages: [{ bolts: "Control valve", nm: 35 }] },
    ],
  },
  {
    id: "turn-twice",
    group: "timing",
    title: "Turn the engine twice and check",
    figure: "front",
    ref: "15 · p. 99, 13 · p. 25",
    say: "With turning-over tool T10531/3 on the tensioning pin (flat side up at TDC) and knurled nut T10531/4, turn the crank two full turns in the direction of engine rotation with a 32 mm open-end wrench.",
    tasks: [
      {
        type: "choice",
        q: "After two turns the painted chain links are no longer on the marks. What does that mean?",
        options: [
          { text: "Nothing: with these sprocket ratios the painted links drift away from the marks once the engine turns", ok: true, why: "The manual notes this. Check timing with the measurement in the next step instead." },
          { text: "The chain has jumped a tooth; take it off again", ok: false, why: "That is expected after turning. Measure the valve timing to be sure." },
        ],
      },
    ],
  },
  {
    id: "valve-timing",
    group: "timing",
    title: "Check the valve timing",
    figure: "front",
    ref: "15 · p. 101–103",
    say: "Find TDC of cylinder 1 with a dial gauge in the spark plug hole (adapter T10170A, gauge set VAS6341), turning only in the direction of engine rotation. If you pass TDC, turn two more full turns. Then measure between the marks on the camshafts.",
    tasks: [
      { type: "measure", spec: "timingIntake", label: "Left outer edge ridge to the intake camshaft mark", items: ["Intake"] },
      { type: "measure", spec: "timingSpread", label: "Exhaust camshaft mark to intake camshaft mark", items: ["Exhaust to intake"] },
      { type: "note", text: "A chain one tooth out shows up as about 6 mm off these values: then the chain must be fitted again." },
    ],
  },
  // ---------------------------------------------------------------- timing covers and damper
  {
    id: "lower-cover",
    group: "covers",
    title: "Fit the new lower timing chain cover",
    figure: "front",
    focus: ["lowerCover"],
    install: ["lowerCover"],
    ref: "15 · p. 67–71",
    say: "Clean the block with a flat scraper and degrease both faces. Both alignment bushings must be in place. The bead goes on the cover's sealing face and on its edges.",
    tasks: [
      {
        type: "choice",
        q: "Nozzle and bead for this cover?",
        options: [
          { text: "Nozzle cut to about 3 mm; bead 2–3 mm; cover on within five minutes", ok: true, why: "And never thicker: excess sealant can get into the lubrication system." },
          { text: "Nozzle 2 mm; bead 1 mm", ok: false, why: "This cover: nozzle about 3 mm, bead 2–3 mm." },
        ],
      },
      { type: "order", diagram: "lowerCover15", when: { lowerCover: "15" }, q: "Click the fifteen cover bolts in order." },
      { type: "order", diagram: "lowerCover8", when: { lowerCover: "8" }, q: "Click the eight cover bolts in order." },
      { type: "torque", what: "Lower cover bolts (new), 15-bolt cover", when: { lowerCover: "15" }, stages: [{ bolts: "Bolts 1–15: in by hand all the way", check: true }, { bolts: "Bolts 1–15", nm: 8 }, { bolts: "Bolts 1, 2, 4, 5 and 7–15", deg: 45 }, { bolts: "Bolts 3 and 6: their 45° waits until the vibration damper is on", check: true }] },
      { type: "torque", what: "Lower cover bolts (new), 8-bolt cover", when: { lowerCover: "8" }, stages: [{ bolts: "Bolts 1–8: in by hand all the way", check: true }, { bolts: "Bolts 1–8", nm: 8 }, { bolts: "Bolts 2, 3 and 5–8", deg: 45 }, { bolts: "Bolts 1 and 4: their 45° waits until the vibration damper is on", check: true }] },
    ],
  },
  {
    id: "damper",
    group: "covers",
    title: "Fit the vibration damper",
    figure: "front",
    focus: ["damper"],
    install: ["damper"],
    ref: "13 · p. 14–17, 24–28, 15 · p. 71–72",
    say: "The damper bolt clamps the damper, the chain sprocket and the crank together. The two cover bolts that wait for their 45° turn come out again to fit the damper, then go back in.",
    tasks: [
      {
        type: "check",
        items: [
          "Damper offered up at TDC, its teeth engaged in the chain sprocket's tooth contour",
          "T10531/3 on the tensioning pin, hex toward the damper; knurled nut T10531/4 tightened until the damper cannot turn",
          "Counterhold T10531/1 fitted with its arrow pointing up; T10531/4 removed, tensioning pin T10531/2 loosened and taken out with T10531/3",
          "New damper bolt with its O-ring oiled, screwed in by hand; T10531/1 removed",
          "Notch on the damper lines up with the arrow on the lower cover (the mark is at about four o'clock)",
        ],
      },
      { type: "torque", what: "Vibration damper bolt (new)", tool: "Counterhold T10355", stages: [{ bolts: "Damper bolt", nm: 150 }, { bolts: "Damper bolt", deg: 90 }] },
      { type: "torque", what: "The two waiting lower cover bolts", when: { lowerCover: "15" }, stages: [{ bolts: "Bolts 3 and 6", deg: 45 }] },
      { type: "torque", what: "The two waiting lower cover bolts", when: { lowerCover: "8" }, stages: [{ bolts: "Bolts 1 and 4", deg: 45 }] },
    ],
  },
  {
    id: "upper-cover",
    group: "covers",
    title: "Fit the upper timing chain cover",
    figure: "front",
    focus: ["upperCover"],
    install: ["upperCover"],
    ref: "15 · p. 66–67, 69",
    say: "New seal and O-ring, both coated with engine oil.",
    tasks: [
      {
        type: "choice",
        q: "Which way round does the cover's seal go?",
        options: [
          { text: "Smaller inner diameter pointing outward", ok: true, why: "That is the seal's installation position." },
          { text: "Larger inner diameter pointing outward", ok: false, why: "The smaller inner diameter points outward." },
        ],
      },
      { type: "order", diagram: "upperCover5", q: "Click the five cover bolts in order." },
      { type: "torque", what: "Upper timing chain cover bolts", stages: [{ bolts: "Bolts 1–5, in sequence", nm: 9 }] },
      { type: "check", items: ["Camshaft adjustment valve 1 (N205) refitted per the manual · 15 · p. 134"] },
    ],
  },
  // ---------------------------------------------------------------- upper oil pan
  {
    id: "upper-pan",
    group: "pan",
    title: "Fit the upper oil pan",
    figure: "pans",
    focus: ["upperPan"],
    install: ["upperPan"],
    ref: "17 · p. 153–154, 158–160",
    say: "The upper pan seals against the block and against the new lower timing chain cover. Dry-fit it first to check the cover is not distorted.",
    tasks: [
      { type: "check", items: ["Old sealant removed from block, pan and lower cover (eyewear on)", "Pan dry-fitted without sealant: gap to the lower cover checked for distortion", "Oil channels in the pan and block clean", "New O-rings fitted"] },
      {
        type: "choice",
        q: "Where does sealant go, besides the pan's own sealing faces?",
        options: [
          { text: "Between the cylinder block and the lower timing chain cover, at the joints", ok: true, why: "Those joints sit under the pan's sealing face." },
          { text: "Nowhere else", ok: false, why: "The manual adds sealant between the block and the lower timing cover." },
        ],
      },
      {
        type: "choice",
        q: "How must the pan sit at the transmission end?",
        options: [
          { text: "Flush with the crankcase", ok: true, why: "The rear crank seal flange seals across both faces, so they must be flush." },
          { text: "A little proud, so the flange compresses it", ok: false, why: "Flush with the crankcase." },
        ],
      },
      { type: "order", diagram: "upperPan13", q: "Click the thirteen pan bolts in order." },
      { type: "torque", what: "Upper oil pan bolts (new); nozzle about 2 mm, bead 2–3 mm, on within five minutes", stages: [{ bolts: "Bolts 1–13: in by hand all the way", check: true }, { bolts: "Bolts 1–13", nm: 15 }, { bolts: "Bolts 1–13", deg: 90 }] },
      { type: "check", items: ["The pan's bolts into the lower timing chain cover tightened like the cover bolts · 17 · p. 160, 15 · p. 67"] },
    ],
  },
];

// Steps after the rear crank seal.
export const AFTER_REAR = [
  // ---------------------------------------------------------------- oil pump and lower pan
  {
    id: "oil-pump",
    group: "lube",
    title: "Install the oil pump",
    figure: "pans",
    focus: ["oilPump"],
    install: ["oilPump"],
    ref: "17 · p. 152, 156–157",
    say: "The pump goes in from below and picks up the drive chain that is already on the crank.",
    tasks: [
      { type: "check", items: ["Both alignment bushings for the pump in place", "New O-rings, lubricated", "Pump's sprocket guided into the drive chain"] },
      {
        type: "choice",
        q: "What must you check on the chain tensioner's spring before removing locking tool T40265?",
        options: [
          { text: "Its wire clip is engaged on the locating tab of the upper oil pan", ok: true, why: "Otherwise the pump chain tension is wrong and the engine can be damaged." },
          { text: "Nothing: the spring finds its own place", ok: false, why: "The wire clip must sit on the locating tab of the upper pan." },
        ],
      },
      { type: "torque", what: "Oil pump bolts", stages: [{ bolts: "Pump bolts", nm: 20 }] },
    ],
  },
  {
    id: "baffle",
    group: "lube",
    title: "Suction line and oil baffle",
    figure: "pans",
    focus: ["suction", "baffle"],
    install: ["suction", "baffle"],
    ref: "17 · p. 152, 154, 157",
    say: "Clean the suction line's screen if there is debris. The baffle is always new: its plastic ribs crush when the lower pan is tightened, so it cannot rattle.",
    tasks: [
      { type: "check", items: ["Suction line screen clean; new O-ring, lubricated", "New oil baffle"] },
      { type: "torque", what: "Suction line and oil baffle bolts", stages: [{ bolts: "Each bolt", nm: 9 }] },
    ],
  },
  {
    id: "lower-pan",
    group: "lube",
    title: "Fit the lower oil pan",
    figure: "pans",
    focus: ["lowerPan"],
    install: ["lowerPan"],
    ref: "17 · p. 153–156",
    say: "Sealant remover, a flat scraper and a rotating plastic brush for the old sealant, eyewear on, then degrease.",
    tasks: [
      {
        type: "choice",
        q: "Why must the bead be no thicker than 2–3 mm here?",
        options: [
          { text: "Excess sealant can enter the pan and clog the oil intake screen", ok: true, why: "Nozzle about 3 mm, bead 2–3 mm, pan on within five minutes." },
          { text: "It doesn't matter for the lower pan", ok: false, why: "It does: squeezed-out sealant can block the suction screen." },
        ],
      },
      { type: "order", diagram: "lowerPan19", q: "Click the nineteen pan bolts in order." },
      { type: "torque", what: "Lower oil pan bolts (new)", stages: [{ bolts: "Bolts 1–19: in by hand all the way", check: true }, { bolts: "Bolts 1–19", nm: 8 }, { bolts: "Bolts 1–19", deg: 45 }] },
      { type: "torque", what: "Oil level sensor G266 (new seal) and drain plug (new)", stages: [{ bolts: "Sensor nuts", nm: 9 }, { bolts: "Drain plug", nm: 30 }] },
      { type: "check", items: ["Sealant left to harden about 30 minutes before any oil goes in"] },
    ],
  },
  // ---------------------------------------------------------------- coolant pump
  {
    id: "coolant-pump",
    group: "coolant",
    title: "Install the coolant pump",
    figure: "coolant",
    focus: ["pump"],
    install: ["pump"],
    ref: "19 · p. 178–181",
    say: "The pump's connection piece plugs into the engine oil cooler, which sits on the sub-assembly bracket. If that bracket isn't on yet, fit the pump when it is.",
    tasks: [
      { type: "check", items: ["New seals and O-rings; O-rings coated with coolant (type per the parts catalog)", "Both centering pins in the block", "Connection piece into the oil cooler; pump pushed onto it and onto the centering pins", "Protective cap removed if the pump is new"] },
      { type: "torque", what: "Coolant pump bolts, in sequence 1–5 shown in the manual", stages: [{ bolts: "Bolts 1–5", nm: 9 }] },
    ],
  },
  {
    id: "coolant-belt",
    group: "coolant",
    title: "Fit the coolant pump toothed belt",
    figure: "coolant",
    focus: ["belt", "wheel"],
    install: ["wheel", "belt", "guard"],
    ref: "19 · p. 179, 181–182",
    say: "The pump is driven by a toothed belt from a drive wheel on the balance shaft. Note the sprocket's installation position.",
    tasks: [
      {
        type: "choice",
        q: "The drive wheel bolt: which way does it tighten?",
        options: [
          { text: "Anticlockwise: it has a left-hand thread", ok: true, why: "The drive wheel bolt is left-hand thread." },
          { text: "Clockwise", ok: false, why: "Left-hand thread: you loosen it clockwise." },
        ],
      },
      { type: "torque", what: "Drive wheel bolt (new, left-hand thread)", stages: [{ bolts: "Bolt", nm: 10 }, { bolts: "Bolt", deg: 90 }] },
      { type: "torque", what: "Toothed belt guard", stages: [{ bolts: "Guard bolts", nm: 9 }] },
    ],
  },
];

// Bolt-order diagrams, positions read off the manual's figures (0–100 box).
export const DIAGRAMS2 = {
  bracket7: { ref: "15 · p. 75", shape: "bracket", bolts: [{ n: 1, x: 52, y: 14 }, { n: 7, x: 10, y: 34 }, { n: 6, x: 92, y: 52 }, { n: 5, x: 14, y: 64 }, { n: 3, x: 30, y: 90 }, { n: 2, x: 52, y: 90 }, { n: 4, x: 74, y: 90 }] },
  lowerCover15: {
    ref: "15 · p. 67",
    shape: "cover",
    bolts: [
      { n: 13, x: 26, y: 8 }, { n: 11, x: 42, y: 6 }, { n: 10, x: 60, y: 6 }, { n: 12, x: 78, y: 12 },
      { n: 9, x: 16, y: 22 }, { n: 8, x: 84, y: 26 }, { n: 7, x: 10, y: 36 }, { n: 6, x: 88, y: 40 },
      { n: 5, x: 12, y: 50 }, { n: 4, x: 88, y: 54 }, { n: 1, x: 10, y: 64 }, { n: 2, x: 86, y: 72 },
      { n: 3, x: 14, y: 80 }, { n: 15, x: 38, y: 92 }, { n: 14, x: 64, y: 92 },
    ],
  },
  lowerCover8: { ref: "15 · p. 68", shape: "cover", bolts: [{ n: 5, x: 30, y: 8 }, { n: 6, x: 70, y: 8 }, { n: 3, x: 10, y: 30 }, { n: 4, x: 90, y: 32 }, { n: 1, x: 12, y: 66 }, { n: 2, x: 88, y: 66 }, { n: 7, x: 40, y: 92 }, { n: 8, x: 60, y: 92 }] },
  upperCover5: { ref: "15 · p. 67", shape: "cover", bolts: [{ n: 4, x: 56, y: 6 }, { n: 2, x: 8, y: 50 }, { n: 1, x: 92, y: 50 }, { n: 5, x: 22, y: 90 }, { n: 3, x: 78, y: 88 }] },
  upperPan13: {
    ref: "17 · p. 154",
    shape: "pan",
    bolts: [
      { n: 12, x: 26, y: 12 }, { n: 10, x: 38, y: 12 }, { n: 8, x: 50, y: 12 }, { n: 6, x: 62, y: 12 }, { n: 4, x: 74, y: 12 },
      { n: 2, x: 94, y: 32 }, { n: 1, x: 90, y: 62 },
      { n: 13, x: 14, y: 86 }, { n: 11, x: 28, y: 86 }, { n: 9, x: 40, y: 86 }, { n: 7, x: 52, y: 86 }, { n: 5, x: 64, y: 86 }, { n: 3, x: 76, y: 86 },
    ],
  },
  lowerPan19: {
    ref: "17 · p. 153",
    shape: "pan",
    bolts: [
      { n: 13, x: 8, y: 8 }, { n: 9, x: 22, y: 8 }, { n: 5, x: 36, y: 8 }, { n: 1, x: 52, y: 8 }, { n: 4, x: 64, y: 8 }, { n: 8, x: 78, y: 8 }, { n: 12, x: 92, y: 10 },
      { n: 17, x: 6, y: 38 }, { n: 19, x: 6, y: 54 }, { n: 18, x: 8, y: 70 },
      { n: 16, x: 94, y: 40 }, { n: 15, x: 92, y: 54 }, { n: 11, x: 88, y: 68 }, { n: 7, x: 82, y: 82 },
      { n: 14, x: 12, y: 94 }, { n: 10, x: 28, y: 94 }, { n: 6, x: 44, y: 94 }, { n: 2, x: 56, y: 94 }, { n: 3, x: 68, y: 94 },
    ],
  },
};
