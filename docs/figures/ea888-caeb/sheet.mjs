// Printable sheets for the EA888 Gen 2 (CAEB) build figure.
// node .claude/skills/anatomy/scripts/print.mjs docs/figures/ea888-caeb/sheet.mjs --format all --png
const piece = (id) => `.it[data-piece="${id}"]`;
const apart = { view: "apart", hide: ["stand"] };
const built = { view: "built", hide: ["stand"] };

export default {
  page: "ea888-caeb-light.html",
  out: "ea888-caeb",
  layers: ".eng-stage > svg",
  rasterSelector: ".eng-stage",
  reduced: true,
  views: {
    apart: { actions: [["key", "Home"], ["wait", 600]] },
    built: { actions: [["key", "End"], ["wait", 600]], reset: [["key", "Home"], ["wait", 600]] },
  },

  name: "EA888 Gen 2",
  title: "EA888 Gen 2 (CAEB): building a 2.0 TFSI",
  number: "EB-001",
  tagline: "2011 Audi A4 · 2.0 TFSI · engine code CAEB",
  question: "What goes into a 2.0 TFSI, and in what order?",
  lead: "A cast-iron block with two balance shafts carries a forged crank, four pistons on 144 mm rods and an aluminium head with roller-finger valvetrain. Chains drive the cams, the balance shafts and the oil pump; a turbocharger and direct injection make 155 kW from 1,984 cc.",
  quote: "Seventeen assemblies, one crankshaft turning them all",
  numbers: [
    ["1,984", "cc", "displacement"],
    ["82.5 × 92.8", "mm", "bore × stroke"],
    ["155", "kW", "211 hp"],
    ["350", "Nm", "peak torque"],
    ["9.6", ": 1", "compression"],
  ],
  spec: [
    ["Layout", "Inline 4, DOHC, 16 valves"],
    ["Displacement", "1,984 cc"],
    ["Bore × stroke", "82.5 × 92.8 mm"],
    ["Rod length", "144 mm"],
    ["Compression", "9.6 : 1"],
    ["Firing order", "1-3-4-2"],
    ["Output", "155 kW (211 hp), 350 Nm"],
    ["Fuel", "Direct injection, cam-driven pump"],
  ],
  why: {
    title: "Why balance shafts?",
    text: "An inline four shakes up and down twice per crank turn, because its pistons move faster in the top half of the stroke than the bottom. Two weighted shafts spinning at twice crank speed in opposite directions cancel that shake.",
  },

  parts: [
    { id: "block", name: "Cylinder block", say: "Cast iron, four 82.5 mm bores at 88 mm centres.", select: piece("block") },
    { id: "balance", name: "Balance shafts", say: "Twice crank speed, opposite ways, to cancel the shake.", select: piece("balance") },
    { id: "crank", name: "Crankshaft and main caps", say: "92.8 mm stroke on five main bearings.", select: piece("crank") },
    { id: "pistons", name: "Pistons and rods", say: "144 mm rods; firing order 1-3-4-2.", select: piece("pistons") },
    { id: "oilpump", name: "Oil pump and chain", say: "Two-stage pump, chain-driven from the crank.", select: piece("oilpump") },
    { id: "head", name: "Cylinder head", say: "16 valves on roller finger followers.", select: piece("head") },
    { id: "cams", name: "Camshafts and adjuster", say: "Half crank speed; intake adjuster, exhaust valvelift.", select: piece("cams") },
    { id: "chains", name: "Timing and balance chains", say: "Crank drives the cams and the balance shafts.", select: piece("chains"), at: [546, 655] },
    { id: "tcover", name: "Timing chain cover", say: "Closes the chain case; carries the cam valve.", select: piece("tcover") },
    { id: "pan", name: "Oil pan", say: "Aluminium upper pan, steel sump, dipstick.", select: piece("pan") },
    { id: "cover", name: "Cylinder head cover", say: "Valvelift actuators, breather, filler cap.", select: piece("cover") },
    { id: "fuel", name: "High-pressure fuel system", say: "Pump on the intake cam, rail, four injectors.", select: piece("fuel") },
    { id: "intake", name: "Intake manifold", say: "Plenum, four runners with flaps, throttle body.", select: piece("intake") },
    { id: "turbo", name: "Turbocharger", say: "Exhaust spins the turbine, which drives the compressor.", select: piece("turbo") },
    { id: "cooling", name: "Oil filter and coolant module", say: "Filter cartridge, thermostat and coolant pipes.", select: piece("cooling") },
    { id: "belt", name: "Belt drive", say: "Damper, alternator, A/C compressor, tensioner.", select: piece("belt") },
    { id: "ignition", name: "Coils and harness", say: "One coil per plug; the harness feeds them.", select: piece("ignition") },
    { id: "stand", select: piece("stand") },
  ],

  main: { ...apart },
  steps: [
    { ...apart, title: "Bottom end", text: "Balance shafts go into the block, then the crankshaft on its five main caps, then the pistons and rods onto the crank pins.", focus: ["block", "balance", "crank", "pistons"] },
    { ...apart, title: "Oil pump and head", text: "The oil pump hangs under the crank on its own chain. The head, with its valves and roller fingers, goes onto the deck.", focus: ["oilpump", "head"] },
    { ...apart, title: "Cams and chains", text: "The camshafts drop into their saddles. Chains from the crank drive the cams at half speed and the balance shafts at twice speed.", focus: ["cams", "chains"] },
    { ...apart, title: "Air and fuel", text: "The high-pressure pump on the intake cam feeds the rail and four injectors; the manifold and turbo bring the air.", focus: ["fuel", "intake", "turbo"] },
    { ...apart, title: "Closing up", text: "Timing cover, oil pan and head cover seal it; then the coolant module, belt drive, coils and harness.", focus: ["tcover", "pan", "cover", "cooling", "belt", "ignition"] },
  ],
  assemblies: [
    { ...apart, title: "Bottom end", text: "Block, balance shafts, crankshaft and the four pistons on their rods: the parts that turn combustion into rotation.", parts: ["block", "balance", "crank", "pistons"] },
    { ...apart, title: "Top end", text: "The head carries 16 valves on roller fingers; the camshafts above open them, intake on one side and exhaust on the other.", parts: ["head", "cams", "cover"] },
    { ...apart, title: "Drives", text: "One crank, three chains: cams at half speed, balance shafts at twice speed, and the oil pump.", parts: ["chains", "oilpump", "tcover"] },
    { ...apart, title: "Air, fuel and spark", text: "Turbo and manifold bring the air, the cam-driven pump and rail inject the fuel, the coils fire the plugs.", parts: ["turbo", "intake", "fuel", "ignition"] },
  ],
  details: [
    { ...apart, title: "Pistons and rods", text: "Four pistons on 144 mm rods. Pins 1 and 4 share a throw, 2 and 3 the opposite one.", around: ["pistons"], focus: ["pistons"] },
    { ...apart, title: "Camshafts", text: "The intake cam carries the vane adjuster at the front and the pump's four-lobe cam at the rear.", around: ["cams"], focus: ["cams"] },
    { ...apart, title: "Turbocharger", text: "Exhaust from all four runners spins the turbine; the wastegate limits boost.", around: ["turbo"], focus: ["turbo"] },
  ],
};
