// The EA888 Gen 2 (CAEB) model: true dimensions in millimetres and the
// kinematics that move every part. `makeModel` has no imports, so the same
// source is inlined into the page with `makeModel.toString()`.
export function makeModel() {
  const DEG = Math.PI / 180;
  const M = {
    name: "EA888 Gen 2 · CAEB",
    displacement: 1984, // cc
    bore: 82.5,
    stroke: 92.8,
    rod: 144, // centre to centre
    compression: 9.6,
    powerKw: 155,
    powerHp: 211,
    torqueNm: 350,
    spacing: 88, // bore centre to bore centre
    firing: [1, 3, 4, 2],
    // Cylinder 1 sits at the timing-chain end (+x).
    cylX: { 1: 132, 2: 44, 3: -44, 4: -132 },
    // Crank throws: 1 and 4 together, 2 and 3 opposite.
    throwDeg: { 1: 0, 2: 180, 3: 180, 4: 0 },
    // Where each cylinder starts its 720° cycle (compression TDC), in crank degrees.
    phaseDeg: { 1: 0, 3: 180, 4: 360, 2: 540 },
    deck: 221, // crank centre to block deck
    compressionHeight: 30.6, // pin centre to crown
    skirt: 26,
    mainR: 26,
    pinR: 23.9,
    camZ: 336,
    camY: { intake: -54, exhaust: 54 },
    valveLift: { intake: 10.7, exhaust: 10 },
  };
  M.r = M.stroke / 2;
  // Crank angle θ (degrees) for a throw: the pin, the piston pin height and the rod's lean.
  M.pinOf = (theta, cyl) => {
    const a = (theta + M.throwDeg[cyl]) * DEG;
    return [M.r * Math.sin(a), M.r * Math.cos(a)];
  };
  M.pistonPin = (theta, cyl) => {
    const [y, z] = M.pinOf(theta, cyl);
    return z + Math.sqrt(M.rod * M.rod - y * y);
  };
  M.crown = (theta, cyl) => M.pistonPin(theta, cyl) + M.compressionHeight;
  // Where cylinder `cyl` is in its four-stroke cycle at crank angle θ (0–720).
  M.cycleOf = (theta, cyl) => (((theta - M.phaseDeg[cyl]) % 720) + 720) % 720;
  M.strokeOf = (theta, cyl) => ["power", "exhaust", "intake", "compression"][Math.floor(M.cycleOf(theta, cyl) / 180)];
  // Valve events in cycle degrees (0 = compression TDC). Lift is a smooth sin² bump.
  M.events = { exhaust: [130, 400], intake: [350, 590] };
  M.lift = (theta, cyl, kind) => {
    const c = M.cycleOf(theta, cyl);
    const [open, close] = M.events[kind];
    if (c < open || c > close) return 0;
    const u = (c - open) / (close - open);
    return M.valveLift[kind] * Math.sin(Math.PI * u) ** 2;
  };
  // The cam lobe's nose points at its follower when the lift peaks.
  M.lobeDeg = (theta, cyl, kind) => {
    const [open, close] = M.events[kind];
    return (theta - M.phaseDeg[cyl] - (open + close) / 2) / 2;
  };
  // Camshafts turn at half crank speed, balance shafts at twice it, the other way.
  M.camDeg = (theta) => theta / 2;
  M.balanceDeg = (theta) => -2 * theta;
  return M;
}
