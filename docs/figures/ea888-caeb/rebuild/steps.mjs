// EA888 Gen 2 (CAEB) rebuild trainer: stage 1, bottom-end assembly.
//
// Every value here comes from the Audi factory repair manual "Servicing – 4-Cylinder
// 1.8L; 2.0L 4V TFSI Engine (EA 888 Generation II)", Edition 06.2019. `ref` gives the
// repair group and the manual's printed page number so each number can be checked
// against the book. Text is paraphrased, not copied. Where the manual leaves a choice
// to the parts catalog or to the engine's version, the step asks the user instead of
// guessing.
//
// Stage 2 (timing drive, covers, oil pans, coolant pump) lives in steps-stage2.mjs and is
// spliced in here in assembly order: the rear crank seal goes on after the upper oil pan.
import { VARIANTS2, SPEC2, GROUPS2, BEFORE_REAR, AFTER_REAR, DIAGRAMS2 } from "./steps-stage2.mjs";

export const VARIANTS = {
  rodBolt: {
    q: "Connecting rod bolt size",
    help: "Measure a rod bolt's thread. The manual gives a different torque for M8 and M9 bolts (Rep. Gr. 13, p. 51).",
    options: [
      { id: "M8", label: "M8" },
      { id: "M9", label: "M9" },
    ],
  },
  oilRing: {
    q: "Oil scraper ring type",
    help: "Your new ring set has either a two-piece or a three-piece oil scraper ring (spring plus two rails). The parts catalog decides which; the specs differ (p. 52–58).",
    options: [
      { id: "2", label: "Two-piece" },
      { id: "3", label: "Three-piece" },
    ],
  },
  flange: {
    q: "Transmission-side sealing flange",
    help: "Count the bolt holes used by the rear crank seal flange and check the bolt material (p. 31).",
    options: [
      { id: "6al", label: "6 bolts, aluminium" },
      { id: "6st", label: "6 bolts, steel" },
      { id: "8", label: "8 bolts" },
    ],
  },
};

// Measurement specs. `nominal` + `dev`: allowed deviation either way. `min`/`max`: new range; `limit`: wear limit (max).
const SPEC = {
  bore: { nominal: 82.51, dev: 0.08, unit: "mm", ref: "13 · p. 58" },
  piston: { nominal: 82.42, dev: 0.04, unit: "mm", ref: "13 · p. 57" },
  mainRadial: { min: 0.017, max: 0.037, limit: 0.15, unit: "mm", ref: "13 · p. 41" },
  mainAxial: { min: 0.07, max: 0.231, limit: 0.3, unit: "mm", ref: "13 · p. 40" },
  rodRadial: { min: 0.02, max: 0.06, limit: 0.09, unit: "mm", ref: "13 · p. 60" },
  rodAxial: { min: 0.1, max: 0.35, limit: 0.4, unit: "mm", ref: "13 · p. 51" },
};
export { SPEC };

const CYL = ["Cyl 1", "Cyl 2", "Cyl 3", "Cyl 4"];
const BORE_POINTS = CYL.flatMap((c) => ["top", "middle", "bottom"].flatMap((h) => [`${c} ${h} A`, `${c} ${h} B`]));

export const GROUPS = [
  { id: "prep", name: "Prepare and measure" },
  { id: "balance", name: "Block and balance shafts" },
  { id: "crank", name: "Crankshaft" },
  { id: "pistons", name: "Pistons and rods" },
  ...GROUPS2.before,
  { id: "rear", name: "Rear crank seal" },
  ...GROUPS2.after,
];

export const STEPS = [
  // ---------------------------------------------------------------- prepare and measure
  {
    id: "variants",
    group: "prep",
    title: "Identify your engine's versions",
    figure: "overview",
    ref: "13 · p. 31, 51–53",
    say: "This manual covers several versions of the Gen 2 engine. Three things change specs in the bottom end, so set them before you start; every later step uses them.",
    tasks: [{ type: "variant", keys: ["rodBolt", "oilRing", "flange"] }],
  },
  {
    id: "parts",
    group: "prep",
    title: "New parts for the bottom end",
    figure: "overview",
    ref: "13 · p. 30–54, 15 · p. 76–77",
    say: "These parts must be new, because the manual says they are replaced whenever they have been removed or loosened. Have them on the bench before you start.",
    tasks: [
      {
        type: "check",
        items: [
          "10 main bearing cap bolts and 6 side bolts (stretch bolts, replace after removing) · 13 · p. 38",
          "8 connecting rod bolts (replace after removing) · 13 · p. 51",
          "Both balance shafts, intake and exhaust side (replace after removing) · 13 · p. 43",
          "Two balance shaft bolts (replace after removing) · 13 · p. 43",
          "Intermediate sprocket and its bolt (never reuse the sprocket) · 15 · p. 76–77",
          "O-ring for the intermediate sprocket's mounting pin · 15 · p. 77",
          "Intake-side balance shaft seal · 13 · p. 49",
          "Transmission-side sealing flange with seal (replace after removing) · 13 · p. 30",
          "8 piston pin circlips (replace after removing) · 13 · p. 52",
          "Sensor wheel and its bolts, only if you remove the wheel from the crank · 13 · p. 38",
          "Bearing shells: replace worn ones; colours come from the letters on the block and crank · 13 · p. 37–40",
          "Sealant for the sealing flange, from the parts catalogue; check its expiry date · 13 · p. 33",
          "Plastigauge, for the bearing clearance checks · 13 · p. 41, 59",
        ],
      },
    ],
  },
  {
    id: "rules",
    group: "prep",
    title: "Workshop rules that apply to every step",
    figure: "overview",
    ref: "00 · p. 5–6",
    say: "Three rules from the general section matter on every bolt you turn.",
    tasks: [
      {
        type: "choice",
        q: "A bolt is specified as \"65 Nm + 90°\". You removed it during teardown. What do you do?",
        options: [
          { text: "Replace it with a new bolt", ok: true, why: "Bolts tightened with an additional turn stretch and must be replaced once removed." },
          { text: "Reuse it if the threads look clean", ok: false, why: "A torque-plus-angle bolt has already yielded. The manual says to replace it every time." },
        ],
      },
      {
        type: "choice",
        q: "Unless a step says otherwise, the torque values in this manual are for…",
        options: [
          { text: "Dry, non-lubricated bolts", ok: true, why: "The manual's specs are for non-lubricated bolts and nuts; a step says so when you must oil a thread (the rod bolts do)." },
          { text: "Oiled threads", ok: false, why: "Oil changes the clamp load at a given torque. Only oil a thread where the step tells you to." },
        ],
      },
      {
        type: "check",
        items: [
          "Unpack new parts only just before fitting them; cover removed parts with lint-free cloths · 00 · p. 5",
          "Plug every open port and channel so nothing falls into the engine · 00 · p. 5",
          "Where no sequence is given, tighten covers and housings diagonally and in steps · 00 · p. 6",
        ],
      },
    ],
  },
  {
    id: "bores",
    group: "prep",
    title: "Measure the cylinder bores",
    figure: "bore",
    ref: "13 · p. 58–59",
    say: "Use a dial bore gauge at three depths, across the engine (A) and along it (B), in a diagonal order. Never hone, drill or grind the bores.",
    tasks: [
      {
        type: "choice",
        q: "Where must the block be when you measure the bores?",
        options: [
          { text: "Off the engine stand", ok: true, why: "The manual warns that readings taken with the block on the engine and transmission holder may be wrong." },
          { text: "Bolted to the engine stand, so it is steady", ok: false, why: "The stand's clamping distorts the readings. Measure the bores before the block goes on the stand." },
        ],
      },
      { type: "measure", spec: "bore", label: "Bore diameter", items: BORE_POINTS },
    ],
  },
  {
    id: "pistons-measure",
    group: "prep",
    title: "Measure the pistons",
    figure: "piston",
    ref: "13 · p. 57",
    say: "Measure about 15 mm up from the lower edge of the skirt, at 90° to the piston pin, with a 75–100 mm micrometer. The 82.42 mm figure is without the skirt coating (about 0.02 mm thick), which wears off.",
    tasks: [{ type: "measure", spec: "piston", label: "Piston diameter", items: CYL }],
  },
  {
    id: "ring-gap",
    group: "prep",
    title: "Check the piston ring gaps",
    figure: "rings",
    ref: "13 · p. 57",
    say: "Push each ring squarely into its bore, about 15 mm from the bottom edge of the cylinder, using a piston without rings, then measure the gap with feeler gauges.",
    tasks: [
      { type: "measure", spec: "gap1", label: "Top compression ring gap", items: CYL },
      { type: "measure", spec: "gap2", label: "Second compression ring gap", items: CYL },
      { type: "measure", spec: "gapOil", label: "Oil scraper ring gap", items: CYL, when: { oilRing: "2" } },
      { type: "note", text: "With the three-piece oil scraper ring, its gap cannot be measured.", when: { oilRing: "3" } },
    ],
  },
  {
    id: "ring-groove",
    group: "prep",
    title: "Check the ring groove clearance",
    figure: "rings",
    ref: "13 · p. 58",
    when: { oilRing: "2" },
    say: "Clean the grooves first, then measure the height clearance of each compression ring in its groove with feeler gauges. With the two-piece oil ring this applies to the compression rings only.",
    tasks: [
      { type: "measure", spec: "groove1", label: "Top ring groove clearance", items: CYL },
      { type: "measure", spec: "groove2", label: "Second ring groove clearance", items: CYL },
    ],
  },
  // ---------------------------------------------------------------- block and balance shafts
  {
    id: "spray-jets",
    group: "balance",
    title: "Fit the piston oil spray jets",
    figure: "block",
    focus: ["jets"],
    install: ["jets"],
    ref: "13 · p. 52, 56",
    say: "One jet per cylinder cools the piston crown, held by a pressure relief valve. The jet for cylinder 4 can only be reached with the crank out, so fit all four now.",
    tasks: [
      {
        type: "choice",
        q: "A spray jet is slightly bent from the teardown. What now?",
        options: [
          { text: "Replace it", ok: true, why: "The manual says never to bend the oil spray jets; a bent jet no longer aims at the piston." },
          { text: "Bend it back into line", ok: false, why: "Never bend the jets. A straightened jet can miss the piston and it will run hot." },
        ],
      },
      { type: "torque", what: "Pressure relief valve (each jet)", tool: "Torx T40 bit", stages: [{ bolts: "Relief valve", nm: 27 }] },
    ],
  },
  {
    id: "bs-pipe",
    group: "balance",
    title: "Fit the exhaust-side balance shaft pipe",
    figure: "balance",
    focus: ["pipe"],
    install: ["pipe"],
    ref: "13 · p. 44, 48",
    say: "The pipe sits under the exhaust-side balance shaft. Its pin must fit into the groove in the block.",
    tasks: [
      {
        type: "choice",
        q: "After you fit the exhaust balance shaft, it does not lie level on the crankcase. Why?",
        options: [
          { text: "The pipe is not seated: its pin is not in the groove", ok: true, why: "The manual says to reinstall the pipe if the shaft is not level." },
          { text: "The bolt needs more torque to pull it down", ok: false, why: "Never pull it down with the bolt. Refit the pipe so its pin engages the groove." },
        ],
      },
    ],
  },
  {
    id: "bs-exhaust",
    group: "balance",
    title: "Install the exhaust-side balance shaft",
    figure: "balance",
    focus: ["bsEx"],
    install: ["bsEx"],
    ref: "13 · p. 43, 47–49",
    say: "Use a new shaft. The clearance to the block is small, so try it in by hand first; if it will not go in without force, chill it.",
    tasks: [
      {
        type: "choice",
        q: "The new shaft will not slide in without force. What does the manual allow?",
        options: [
          { text: "30 minutes in a freezer, or cooling spray, then fit it", ok: true, why: "Cooling shrinks the shaft enough to slide in." },
          { text: "Tap it in with a soft mallet", ok: false, why: "Never force it. Cool it instead." },
          { text: "Heat the block around the bore", ok: false, why: "The manual only describes cooling the shaft." },
        ],
      },
      { type: "check", items: ["Bearings lubricated with engine oil", "Shaft lies level on the crankcase before the bolt is tightened"] },
      { type: "torque", what: "Balance shaft bolt, exhaust side (new)", stages: [{ bolts: "Bolt", nm: 9 }] },
    ],
  },
  {
    id: "bs-intake",
    group: "balance",
    title: "Install the intake-side balance shaft",
    figure: "balance",
    focus: ["bsIn"],
    install: ["bsIn"],
    ref: "13 · p. 43, 46",
    say: "Same as the exhaust side: a new shaft, chilled if it will not go in by hand, bearings oiled.",
    tasks: [
      { type: "check", items: ["New intake-side shaft (replace after removing)", "Bearings lubricated with engine oil", "Cooled first if it would not slide in without force"] },
      { type: "torque", what: "Balance shaft bolt, intake side (new)", stages: [{ bolts: "Bolt", nm: 9 }] },
    ],
  },
  {
    id: "idler-pin",
    group: "balance",
    title: "Fit the intermediate sprocket's mounting pin",
    figure: "balance",
    focus: ["pin"],
    install: ["pin"],
    ref: "13 · p. 46, 15 · p. 77",
    say: "The mounting (bearing) pin carries the intermediate sprocket between the crank and the intake balance shaft.",
    tasks: [
      { type: "check", items: ["New O-ring, coated with engine oil", "Mounting pin coated with engine oil", "Its alignment pin engages the hole in the cylinder block"] },
    ],
  },
  {
    id: "idler",
    group: "balance",
    title: "Install the intermediate sprocket",
    figure: "balance",
    focus: ["idler"],
    install: ["idler"],
    ref: "13 · p. 46–47, 15 · p. 76–77",
    say: "A new sprocket has an anti-friction coating that wears in and sets the backlash. Reusing an old one leaves no backlash and can destroy the engine.",
    tasks: [
      {
        type: "choice",
        q: "How do you time the intermediate sprocket to the balance shaft?",
        options: [
          { text: "Paint-mark a tooth flank on the sprocket; the balance shaft's mark must sit between the marks", ok: true, why: "That is the manual's method, and you check the marks again after tightening." },
          { text: "Any position: the chain sets the timing later", ok: false, why: "The sprocket meshes directly with the balance shaft gear. Its marks set the shaft's phase." },
        ],
      },
      {
        type: "torque",
        what: "Intermediate sprocket bolt (new)",
        stages: [
          { bolts: "Bolt", nm: 10 },
          { bolts: "Turn the sprocket: it must have no play; if it has, loosen and tighten again", check: true },
          { bolts: "Bolt", nm: 25 },
          { bolts: "Bolt", deg: 90 },
        ],
      },
      { type: "check", items: ["Marks on the sprocket and balance shaft still line up after tightening"] },
    ],
  },
  {
    id: "bs-seal",
    group: "balance",
    title: "Fit the intake-side balance shaft seal",
    figure: "balance",
    focus: ["seal"],
    install: ["seal"],
    ref: "13 · p. 49–50",
    say: "The seal sits where the balance shaft carries the coolant pump's toothed-belt drive wheel; the wheel goes on later, with the coolant pump. (The manual calls this seal intake side in one place and exhaust side in another; it is the one behind that drive wheel.)",
    tasks: [
      {
        type: "choice",
        q: "What goes on the balance shaft's sealing surface before you push the seal on?",
        options: [
          { text: "Transmission fluid", ok: true, why: "The manual specifies transmission fluid on this seal's running surface." },
          { text: "Sealant", ok: false, why: "Not sealant: this is a lip seal on a turning shaft." },
          { text: "Nothing, keep it dry", ok: false, why: "The manual says to coat the surface with transmission fluid." },
        ],
      },
      { type: "check", items: ["The word \"Luftseite\" (\"Outside\") on the seal can be read from outside", "Pressed in square with the seal installer, not tilted"] },
    ],
  },
  // ---------------------------------------------------------------- crankshaft
  {
    id: "shells-upper",
    group: "crank",
    title: "Fit the upper main bearing shells",
    figure: "mains",
    focus: ["shellUp"],
    install: ["shellUp"],
    ref: "13 · p. 37, 39–40",
    say: "Shell thickness is graded by colour. The letters on the block (on the oil pan face or the transmission end) give the colour of each upper shell, one letter per bearing starting at bearing 1.",
    tasks: [
      {
        type: "choice",
        q: "Which shells go into the cylinder block?",
        options: [
          { text: "The shells with the oil groove", ok: true, why: "Block shells have the oil groove; cap shells have none." },
          { text: "The shells without the oil groove", ok: false, why: "Those are the bearing cap shells." },
        ],
      },
      {
        type: "choice",
        q: "The block is stamped \"R\" for bearing 2. Which colour shell goes there?",
        options: [
          { text: "Red", ok: true, why: "Block letters: G yellow, R red, S black, VI purple, WS white. Confirm the allocation in the parts catalogue." },
          { text: "Black", ok: false, why: "Black is S on the block." },
          { text: "Yellow", ok: false, why: "Yellow is G on the block." },
        ],
      },
      { type: "check", items: ["Each shell's tab sits in its notch", "Shells oiled with engine oil"] },
    ],
  },
  {
    id: "thrust",
    group: "crank",
    title: "Fit the thrust washers",
    figure: "mains",
    focus: ["thrust"],
    install: ["thrust"],
    ref: "13 · p. 38",
    say: "The thrust washers locate the crank end to end.",
    tasks: [
      {
        type: "choice",
        q: "At which main bearing do the thrust washers go?",
        options: [
          { text: "Bearing 3, the middle one", ok: true, why: "The thrust washers are for bearing 3." },
          { text: "Bearing 1, at the belt pulley end", ok: false, why: "Bearing 3 carries the thrust washers." },
          { text: "Bearing 5, at the transmission end", ok: false, why: "Bearing 3 carries the thrust washers." },
        ],
      },
      { type: "check", items: ["Thrust washers lubricated with engine oil"] },
    ],
  },
  {
    id: "sensor-wheel",
    group: "crank",
    title: "Sensor wheel on the crankshaft",
    figure: "mains",
    focus: ["crank"],
    ref: "13 · p. 38, 41–42",
    say: "The sensor wheel for the engine speed sensor (G28) is bolted to the crank. Leave it on if you can: once its bolts have been loosened, the wheel and the bolts must be replaced.",
    tasks: [
      {
        type: "choice",
        q: "You loosened the sensor wheel bolts during teardown. What does the manual require?",
        options: [
          { text: "New sensor wheel, new bolts, and an adaptation with a diagnostic tester afterwards", ok: true, why: "The countersinks deform on the second tightening, so the wheel would sit loose under the bolt heads." },
          { text: "New bolts only", ok: false, why: "The wheel itself must be replaced too, and the adaptation done afterwards." },
        ],
      },
      { type: "torque", what: "Sensor wheel bolts (only if replaced; new)", note: "The wheel fits in one position only: the holes are offset.", stages: [{ bolts: "Each bolt", nm: 10 }, { bolts: "Each bolt", deg: 90 }] },
    ],
  },
  {
    id: "crank-in",
    group: "crank",
    title: "Lay in the crankshaft and fit the lower shells",
    figure: "mains",
    focus: ["crank", "shellLow"],
    install: ["crank", "shellLow"],
    ref: "13 · p. 37–40",
    say: "Lay the crank in the oiled upper shells. The letters on the crankshaft give the colour of each lower (cap) shell, one letter per bearing.",
    tasks: [
      {
        type: "choice",
        q: "The crank is stamped \"B\" for bearing 4. Which cap shell?",
        options: [
          { text: "Blue", ok: true, why: "Crank letters: R red, G yellow, B blue, W white, V purple, S black. Confirm in the parts catalogue." },
          { text: "Black", ok: false, why: "Black is S." },
        ],
      },
      { type: "check", items: ["Cap shells are the ones without the oil groove", "The tabs of the block shell and cap shell lie above one another", "Do not rest the crank on its sensor wheel when you set it down"] },
    ],
  },
  {
    id: "main-plastigauge",
    group: "crank",
    title: "Check main bearing clearance with Plastigauge",
    figure: "plastigauge",
    ref: "13 · p. 41",
    say: "Measure with the old bolts, then fit new bolts for final assembly. Lay the Plastigauge across the full bearing width, centred on the shell, fit the cap with the full tightening sequence, and do not turn the crank. Remove the cap and read the squashed strip against the scale.",
    tasks: [
      {
        type: "choice",
        q: "Which bolts do you use for this check?",
        options: [
          { text: "The old bolts", ok: true, why: "The manual uses the old bolts for clearance checks and new ones for the final assembly." },
          { text: "The new bolts, then reuse them", ok: false, why: "Tightening the new bolts to angle would stretch them before final assembly." },
        ],
      },
      { type: "measure", spec: "mainRadial", label: "Main bearing radial clearance", items: ["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4", "Bearing 5"] },
    ],
  },
  {
    id: "main-caps",
    group: "crank",
    title: "Fit the main bearing caps",
    figure: "mains",
    focus: ["caps"],
    install: ["caps"],
    ref: "13 · p. 38–39",
    say: "Clean the Plastigauge off, oil the shells and fit each cap in the place you marked at teardown, with new bolts.",
    tasks: [
      {
        type: "choice",
        q: "Where does bearing cap 1 go?",
        options: [
          { text: "At the belt pulley end", ok: true, why: "Cap 1 is the belt pulley side; the caps were marked 1–5 at teardown." },
          { text: "At the transmission end", ok: false, why: "Cap 1 is at the belt pulley end." },
        ],
      },
      {
        type: "choice",
        q: "Which caps also have bolts from the side of the block?",
        options: [
          { text: "Caps 2, 3 and 4", ok: true, why: "Those three caps carry the six side bolts (the \"A\" arrows in the manual's figure)." },
          { text: "All five caps", ok: false, why: "Only caps 2, 3 and 4." },
        ],
      },
    ],
  },
  {
    id: "main-torque",
    group: "crank",
    title: "Tighten the main bearing caps",
    figure: "mains",
    focus: ["bolts"],
    install: ["bolts"],
    ref: "13 · p. 38",
    say: "Five steps, in order: all bolts in by hand, the ten cap bolts to torque and then angle in sequence, then the six side bolts.",
    tasks: [
      { type: "order", diagram: "mains", q: "Click the ten cap bolts in the order you tighten them." },
      {
        type: "torque",
        what: "Main bearing cap bolts (new)",
        stages: [
          { bolts: "Bolts 1–10 and the six side bolts: screw in by hand all the way", check: true },
          { bolts: "Bolts 1–10, in sequence", nm: 65 },
          { bolts: "Bolts 1–10, in sequence", deg: 90 },
          { bolts: "Six side bolts", nm: 15 },
          { bolts: "Six side bolts", deg: 90 },
        ],
      },
    ],
  },
  {
    id: "main-axial",
    group: "crank",
    title: "Check crankshaft end play",
    figure: "endplay",
    ref: "13 · p. 40",
    say: "Mount a dial gauge on the block with its holder, against a crank counterweight. Push the crank toward the gauge by hand and zero it, then push it away and read.",
    tasks: [{ type: "measure", spec: "mainAxial", label: "Crankshaft axial clearance", items: ["Crankshaft"] }],
  },
  // ---------------------------------------------------------------- pistons and rods
  {
    id: "rods-new",
    group: "pistons",
    title: "Prepare the connecting rods",
    figure: "rod",
    ref: "13 · p. 51–52, 59",
    say: "Cracked rods: each cap fits only its own rod, one way round. Rods are replaced as a set.",
    tasks: [
      {
        type: "choice",
        q: "A new rod's cap will not come off by hand. What does the manual describe?",
        options: [
          { text: "Clamp it lightly in soft jaws below the marked line, back the bolts out about five turns, tap the cap free with a plastic hammer", ok: true, why: "New rods are sometimes not fully separated at the break line." },
          { text: "Pry it apart with a screwdriver", ok: false, why: "That damages the cracked faces, which must mate perfectly." },
        ],
      },
      { type: "check", items: ["Each rod and cap colour-marked with its cylinder", "Rods replaced as a complete set if any is replaced"] },
    ],
  },
  {
    id: "piston-rod",
    group: "pistons",
    title: "Assemble pistons to rods",
    figure: "piston",
    focus: ["piston", "ppin", "clips", "rod"],
    install: ["ppin", "clips"],
    ref: "13 · p. 52, 54",
    say: "Oil each piston pin and fit it with new circlips. Get the direction right now: you cannot fix it once the piston is in the bore.",
    tasks: [
      {
        type: "choice",
        q: "Which way does the arrow on the piston crown point?",
        options: [
          { text: "Toward the belt pulley end", ok: true, why: "The arrow on the crown points to the belt pulley side." },
          { text: "Toward the transmission", ok: false, why: "It points to the belt pulley side." },
        ],
      },
      {
        type: "choice",
        q: "And the marks on the rod and cap?",
        options: [
          { text: "Also toward the belt pulley end", ok: true, why: "Rod marks face the belt pulley side, the same way as the piston arrow." },
          { text: "Toward the transmission end", ok: false, why: "The rod marks face the belt pulley side." },
        ],
      },
      { type: "check", items: ["Piston pin coated with engine oil", "Two new circlips per piston, fully seated", "Each piston matched to its cylinder"] },
    ],
  },
  {
    id: "rings",
    group: "pistons",
    title: "Fit the piston rings",
    figure: "rings",
    focus: ["rings", "ringsBack"],
    install: ["rings", "ringsBack"],
    ref: "13 · p. 52–53",
    say: "Fit the compression rings with ring pliers, opening them only far enough to pass over the piston. \"TOP\" faces the crown.",
    tasks: [
      {
        type: "choice",
        when: { oilRing: "3" },
        q: "In what order do the three-piece oil scraper ring's parts go in?",
        options: [
          { text: "Spring first, then the lower rail, then the upper rail, by hand", ok: true, why: "Spring, lower plate, upper plate, by hand. The rails' joints and the spring's are offset 90°." },
          { text: "Both rails, then the spring between them", ok: false, why: "The spring goes in the groove first." },
        ],
      },
      {
        type: "choice",
        when: { oilRing: "3" },
        q: "With the three-piece oil ring, how do you tell the two compression rings apart?",
        options: [
          { text: "Top ring: bevel on the upper inside edge. Second ring: step on the lower outside edge. TOP up on both", ok: true, why: "That is the manual's allocation for this ring set." },
          { text: "They are the same; fit either way up", ok: false, why: "They are different rings and each has an up side." },
        ],
      },
      {
        type: "choice",
        when: { oilRing: "2" },
        q: "With the two-piece oil ring, which way up do the rings go?",
        options: [
          { text: "\"TOP\" or \"R\" toward the crown, on every ring", ok: true, why: "The compression rings and the oil ring's marking all face the crown." },
          { text: "Markings down", ok: false, why: "The markings face the piston crown." },
        ],
      },
      {
        type: "choice",
        q: "How are the ring gaps arranged round the piston?",
        options: [
          { text: "Offset 120° from each other", ok: true, why: "Gaps are offset 120°, and the oil ring's (upper) joint 120° from the neighbouring compression ring." },
          { text: "All lined up, so they are easy to check", ok: false, why: "Lined-up gaps let gas blow by. Offset them 120°." },
          { text: "180° apart", ok: false, why: "The manual specifies 120°." },
        ],
      },
    ],
  },
  {
    id: "rod-shells",
    group: "pistons",
    title: "Fit the rod bearing shells",
    figure: "rod",
    focus: ["rodShells"],
    install: ["rodShells"],
    ref: "13 · p. 51, 54",
    say: "Place each shell centrally in the rod and in the cap, with the same distance (dimension a = a) at both ends.",
    tasks: [{ type: "check", items: ["Shells centred in rod and cap (a = a)", "Worn shells replaced", "Shell contact faces coated with engine oil"] }],
  },
  {
    id: "rod-plastigauge",
    group: "pistons",
    title: "Check rod bearing clearance with Plastigauge",
    figure: "plastigauge",
    ref: "13 · p. 59–60",
    say: "Like the mains: old bolts, Plastigauge across the full width, fit the cap to full torque without turning the crank, then read the strip.",
    tasks: [{ type: "measure", spec: "rodRadial", label: "Rod bearing radial clearance", items: CYL }],
  },
  {
    id: "pistons-in",
    group: "pistons",
    title: "Install the pistons",
    figure: "piston",
    focus: ["piston", "rod"],
    ref: "13 · p. 54–55",
    say: "Oil the shell faces and the cylinder wall, then push each piston in carefully by hand through the piston installation sleeve.",
    tasks: [
      { type: "check", items: ["Bearing shells and cylinder wall oiled", "Piston pushed in by hand through the sleeve (T40347), not hammered", "Arrow on the crown toward the belt pulley end", "Each piston in its own cylinder"] },
    ],
  },
  {
    id: "rod-caps",
    group: "pistons",
    title: "Fit and tighten the rod caps",
    figure: "rod",
    focus: ["rodCap", "rodBolts"],
    install: ["rodCap", "rodBolts"],
    ref: "13 · p. 51",
    say: "Each cap goes on its own rod, marks toward the belt pulley end. New bolts, with oil on the threads and under the heads: this is one of the places the manual asks for lubricated bolts.",
    tasks: [
      { type: "check", items: ["Cap matches its rod's colour mark", "Marks face the belt pulley end", "New bolts, threads and contact faces oiled"] },
      { type: "torque", what: "Connecting rod bolts, M8 (new, oiled)", when: { rodBolt: "M8" }, stages: [{ bolts: "Both bolts", nm: 30 }, { bolts: "Both bolts", deg: 90 }] },
      { type: "torque", what: "Connecting rod bolts, M9 (new, oiled)", when: { rodBolt: "M9" }, stages: [{ bolts: "Both bolts", nm: 45 }, { bolts: "Both bolts", deg: 90 }] },
    ],
  },
  {
    id: "rod-axial",
    group: "pistons",
    title: "Check rod side clearance",
    figure: "rod",
    ref: "13 · p. 51",
    say: "Measure the axial (side) clearance of each rod on its crank pin with feeler gauges.",
    tasks: [{ type: "measure", spec: "rodAxial", label: "Rod axial clearance", items: CYL }],
  },
  ...BEFORE_REAR,
  // ---------------------------------------------------------------- rear crank seal (after the upper oil pan)
  {
    id: "flange-prep",
    group: "rear",
    title: "Prepare for the transmission-side sealing flange",
    figure: "flange",
    focus: ["flange"],
    ref: "13 · p. 33",
    say: "A new flange with its seal, sealant on the flange face. Prepare everything first: once the bead is on you have five minutes.",
    tasks: [
      { type: "check", items: ["Protective eyewear on", "Clean cloth over the open oil pan so no sealant residue falls in", "Old sealant removed from the block, for example with a rotating plastic brush", "Sealing faces and crank journal free of oil and grease", "Sealant within its expiry date; nozzle cut at the front mark (about 2 mm)"] },
      {
        type: "choice",
        q: "Do you oil the new seal's lip before fitting it?",
        options: [
          { text: "No: no oil on this seal, and wipe the crank journal clean", ok: true, why: "The manual says not to oil this seal and to remove oil residue from the journal." },
          { text: "Yes, a little engine oil", ok: false, why: "Not on this seal: fit it dry on a clean journal." },
        ],
      },
    ],
  },
  {
    id: "flange-fit",
    group: "rear",
    title: "Fit the sealing flange",
    figure: "flange",
    focus: ["flange"],
    install: ["flange"],
    ref: "13 · p. 34–35",
    say: "Run the bead, then use the seal installer guide pieces to slide the flange over the crank journal so the lip cannot fold over.",
    tasks: [
      {
        type: "choice",
        q: "How thick is the sealant bead?",
        options: [
          { text: "2 to 3 mm", ok: true, why: "Thicker is not better: excess sealant can get into the lubrication system." },
          { text: "As thick as you can, to be safe", ok: false, why: "Excess sealant can block oil passages. 2–3 mm." },
          { text: "1 mm or less", ok: false, why: "The manual specifies 2–3 mm." },
        ],
      },
      {
        type: "choice",
        q: "After fitting, which way must the sealing lip point?",
        options: [
          { text: "Inward, toward the engine", ok: true, why: "If the lip folds outward during fitting, it will leak." },
          { text: "Outward, toward the transmission", ok: false, why: "A lip folded outward leaks. It must point toward the engine." },
        ],
      },
      { type: "check", items: ["Flange fitted within five minutes of applying the sealant", "Guide pieces T10122/6A and T10122/1 used, not tilted and clean"] },
    ],
  },
  {
    id: "flange-torque",
    group: "rear",
    title: "Tighten the sealing flange",
    figure: "flange",
    focus: ["flange"],
    ref: "13 · p. 31, 35",
    say: "The sequence and specification depend on your flange version.",
    tasks: [
      { type: "order", diagram: "flange6", when: { flange: ["6al", "6st"] }, q: "Click the six bolts in tightening order." },
      { type: "order", diagram: "flange8", when: { flange: "8" }, q: "Click the eight bolts in tightening order." },
      { type: "torque", what: "Sealing flange bolts, aluminium (new)", when: { flange: "6al" }, stages: [{ bolts: "Bolts 1–6: screw in by hand all the way", check: true }, { bolts: "Bolts 1–6", nm: 4 }, { bolts: "Bolts 1–6", deg: 45 }] },
      { type: "torque", what: "Sealing flange bolts, steel", when: { flange: "6st" }, stages: [{ bolts: "Bolts 1–6: screw in by hand all the way", check: true }, { bolts: "Bolts 1–6", nm: 9 }] },
      { type: "torque", what: "Sealing flange bolts", when: { flange: "8" }, stages: [{ bolts: "Bolts 1–8: screw in by hand all the way", check: true }, { bolts: "Bolts 1–8", nm: 9 }] },
      {
        type: "choice",
        q: "When can engine oil go in?",
        options: [
          { text: "After the sealant has cured for about 30 minutes", ok: true, why: "The sealant needs about 30 minutes to harden before the engine is filled." },
          { text: "Straight away", ok: false, why: "Wait about 30 minutes for the sealant to harden." },
        ],
      },
    ],
  },
  ...AFTER_REAR,
];
Object.assign(VARIANTS, VARIANTS2);
Object.assign(SPEC, SPEC2);

// Ring gap and groove specs depend on the oil ring version.
export function specFor(id, variants) {
  if (SPEC[id]) return SPEC[id];
  const two = variants.oilRing === "2";
  const table = {
    gap1: two ? { min: 0.2, max: 0.4, limit: 0.8 } : { min: 0.3, max: 0.4, limit: 0.6 },
    gap2: two ? { min: 0.2, max: 0.4, limit: 0.8 } : { min: 0.4, max: 0.5, limit: 0.7 },
    gapOil: { min: 0.25, max: 0.5, limit: 0.8 },
    groove1: { min: 0.06, max: 0.09, limit: 0.2 },
    groove2: { min: 0.03, max: 0.06, limit: 0.15 },
  }[id];
  return { ...table, unit: "mm", ref: id.startsWith("gap") ? "13 · p. 57" : "13 · p. 58" };
}

// Bolt-order diagrams, drawn as in the manual's figures. `seq` is the bolt number at each
// position; positions are in a 0–100 box.
export const DIAGRAMS = {
  // Main caps seen from the oil pan side, as in the manual's figure on p. 38: the middle cap
  // takes 1–2, its neighbours 3–4 and 5–6, the end caps 7–8 and 9–10.
  mains: {
    ref: "13 · p. 38",
    caps: 5,
    bolts: [
      { n: 7, cap: 0, side: 0 }, { n: 8, cap: 0, side: 1 },
      { n: 3, cap: 1, side: 0 }, { n: 4, cap: 1, side: 1 },
      { n: 1, cap: 2, side: 0 }, { n: 2, cap: 2, side: 1 },
      { n: 5, cap: 3, side: 0 }, { n: 6, cap: 3, side: 1 },
      { n: 9, cap: 4, side: 0 }, { n: 10, cap: 4, side: 1 },
    ],
    side: [1, 2, 3],
  },
  // Rear flange, seen from the transmission side (p. 31). Positions as in the figure.
  flange6: {
    ref: "13 · p. 31",
    bolts: [
      { n: 4, x: 70, y: 12 }, { n: 3, x: 22, y: 30 }, { n: 1, x: 84, y: 62 },
      { n: 2, x: 16, y: 70 }, { n: 6, x: 34, y: 92 }, { n: 5, x: 66, y: 92 },
    ],
  },
  flange8: {
    ref: "13 · p. 31",
    bolts: [
      { n: 5, x: 30, y: 12 }, { n: 7, x: 70, y: 12 }, { n: 1, x: 12, y: 46 }, { n: 4, x: 88, y: 46 },
      { n: 3, x: 14, y: 72 }, { n: 2, x: 86, y: 72 }, { n: 8, x: 38, y: 94 }, { n: 6, x: 62, y: 94 },
    ],
  },
};
Object.assign(DIAGRAMS, DIAGRAMS2);
