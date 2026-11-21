// Symptom guide: what the driver notices → what to check first → likely causes → jobs.
// `stop` marks symptoms where the car should not be driven until checked.
// Causes are general and commonly reported ones for the Audi A4 (B8) 2.0 TFSI, not a diagnosis.
export const AREAS = ["Engine", "Cooling", "Electrics", "Brakes", "Suspension and steering", "Wheels and tires", "Transmission", "Comfort and visibility"];

export const SYMPTOMS = [
  {
    id: "oil-light", area: "Engine", stop: true,
    symptom: "Red oil pressure warning light",
    first: ["Stop the engine as soon as it's safe: running without oil pressure destroys the engine in minutes", "Check the oil level on level ground with the engine off"],
    causes: [
      { text: "Oil level too low", jobs: ["a4-b8-fluid-check", "a4-b8-oil-change"] },
      { text: "Low oil pressure from wear, a blocked pickup or a faulty switch: test the pressure", jobs: ["ea888-compression-oil-pressure"] },
    ],
  },
  {
    id: "overheat", area: "Cooling", stop: true,
    symptom: "Temperature gauge high, or coolant warning light",
    first: ["Stop and switch off: driving an overheating engine can warp the head", "Let it cool fully before opening the coolant cap"],
    causes: [
      { text: "Coolant level low from a leak, commonly the coolant pump and thermostat housing", jobs: ["ea888-coolant-pump-leak", "a4-b8-fluid-check"] },
      { text: "Air trapped after a coolant change", jobs: ["a4-b8-coolant-change"] },
    ],
  },
  {
    id: "coolant-drop", area: "Cooling",
    symptom: "Coolant level keeps dropping, or a sweet smell after driving",
    first: ["Check the level with the engine cold and note how fast it drops", "Look for pink or purple crust under the intake side of the engine"],
    causes: [
      { text: "Coolant pump and thermostat housing leak (commonly reported)", jobs: ["ea888-coolant-pump-leak"] },
      { text: "Hose, radiator or expansion tank leak: find it with a pressure test", jobs: ["ea888-coolant-pump-leak"] },
    ],
  },
  {
    id: "cold-rattle", area: "Engine",
    symptom: "Rattle from the engine for a few seconds after a cold start",
    first: ["Read the fault codes: cam/crank correlation codes make it more urgent", "Avoid high revs until it's checked"],
    causes: [{ text: "Timing chain stretch or a worn tensioner (commonly reported)", jobs: ["ea888-timing-chain-check"] }],
  },
  {
    id: "misfire", area: "Engine",
    symptom: "Engine warning light, misfire, shaking or rough idle",
    first: ["Read the fault codes: misfire codes name the cylinder", "A flashing engine light means a severe misfire: drive gently and get it checked soon"],
    causes: [
      { text: "Worn spark plugs or a failing ignition coil", jobs: ["a4-b8-spark-plugs"] },
      { text: "Failed crankcase breather (whistle at idle)", jobs: ["ea888-pcv-oil-separator"] },
      { text: "Carbon on the intake valves, especially on cold starts", jobs: ["a4-b8-carbon-cleaning"] },
      { text: "Low compression in one cylinder", jobs: ["ea888-compression-oil-pressure"] },
    ],
  },
  {
    id: "whistle", area: "Engine",
    symptom: "Whistle or squeal at idle; oil filler cap hard to remove with the engine running",
    first: ["Try the oil filler test with the engine idling"],
    causes: [{ text: "Failed crankcase breather (PCV) diaphragm", jobs: ["ea888-pcv-oil-separator"] }],
  },
  {
    id: "oil-use", area: "Engine",
    symptom: "Needs oil topping up between services",
    first: ["Measure how much it uses over 1,000 miles", "Look for oil leaks under the car"],
    causes: [
      { text: "Worn or stuck piston rings (commonly reported on early CAEB engines)", jobs: ["ea888-compression-oil-pressure"] },
      { text: "Failed crankcase breather", jobs: ["ea888-pcv-oil-separator"] },
    ],
  },
  {
    id: "power", area: "Engine",
    symptom: "Loss of power or hesitation when accelerating",
    first: ["Read the fault codes", "Note whether it happens cold, hot or all the time"],
    causes: [
      { text: "Carbon on the intake valves", jobs: ["a4-b8-carbon-cleaning"] },
      { text: "Spark plugs or ignition coils", jobs: ["a4-b8-spark-plugs"] },
      { text: "Clogged air filter", jobs: ["a4-b8-air-filter"] },
    ],
  },
  {
    id: "belt-squeal", area: "Engine",
    symptom: "Squeal from the engine bay at start-up or with the A/C on",
    first: ["Look at the auxiliary belt for cracks and glazing"],
    causes: [{ text: "Worn auxiliary belt or weak tensioner", jobs: ["a4-b8-aux-belt"] }],
  },
  {
    id: "battery-light", area: "Electrics",
    symptom: "Battery warning light while driving",
    first: ["Check the auxiliary belt is still on and turning the alternator", "Avoid switching the engine off far from home: it may not restart"],
    causes: [
      { text: "Broken or slipping auxiliary belt", jobs: ["a4-b8-aux-belt"] },
      { text: "Battery or charging fault: test the battery", jobs: ["a4-b8-battery"] },
    ],
  },
  {
    id: "no-start", area: "Electrics",
    symptom: "Cranks slowly, clicks, or won't start",
    first: ["Look at the dashboard lights when you turn the key: dim or flickering lights point to the battery"],
    causes: [
      { text: "Flat or worn-out battery", jobs: ["a4-b8-jump-start", "a4-b8-battery"] },
    ],
  },
  {
    id: "brake-pedal", area: "Brakes", stop: true,
    symptom: "Brake pedal soft, spongy or sinks to the floor",
    first: ["Don't drive: check the brake fluid level and look for leaks at each wheel"],
    causes: [
      { text: "Low fluid from a leak, or air in the brakes", jobs: ["a4-b8-fluid-check", "a4-b8-brake-fluid"] },
      { text: "Old fluid that has absorbed water (soft pedal after hard use)", jobs: ["a4-b8-brake-fluid"] },
    ],
  },
  {
    id: "brake-shake", area: "Brakes",
    symptom: "Steering wheel shakes when braking",
    first: ["Note whether it shakes from high speed only or at all speeds"],
    causes: [
      { text: "Front discs worn unevenly (often from rust on the hub face or overtightened wheel bolts)", jobs: ["a4-b8-front-brakes"] },
      { text: "Rear discs, if the whole car shudders", jobs: ["a4-b8-rear-brakes"] },
    ],
  },
  {
    id: "brake-noise", area: "Brakes",
    symptom: "Squeal or grinding when braking, or the brake pad warning light",
    first: ["Look at the pad thickness through the wheel spokes", "Grinding means metal on metal: stop driving and replace the pads and discs"],
    causes: [
      { text: "Front pads worn", jobs: ["a4-b8-front-brakes"] },
      { text: "Rear pads worn", jobs: ["a4-b8-rear-brakes"] },
    ],
  },
  {
    id: "front-knock", area: "Suspension and steering",
    symptom: "Knock or clunk from the front over bumps",
    first: ["Note whether it's over big bumps, small bumps, or when turning"],
    causes: [
      { text: "Worn upper control arms (commonly reported on the B8)", jobs: ["a4-b8-upper-control-arms"] },
      { text: "Worn anti-roll bar links (more of a rattle over small bumps)", jobs: ["a4-b8-sway-bar-links"] },
      { text: "Worn shock absorber top mount (knock when turning at low speed)", jobs: ["a4-b8-shocks"] },
    ],
  },
  {
    id: "bouncy", area: "Suspension and steering",
    symptom: "Car bounces or floats after bumps, or dives hard when braking",
    first: ["Look for oil on the shock bodies; do the bounce test"],
    causes: [{ text: "Worn shock absorbers", jobs: ["a4-b8-shocks"] }],
  },
  {
    id: "pulls", area: "Wheels and tires",
    symptom: "Pulls to one side, or tires wear on one edge",
    first: ["Check the tire pressures, cold"],
    causes: [
      { text: "Wrong or uneven tire pressures", jobs: ["a4-b8-wheel-swap"] },
      { text: "Worn control arms, then alignment out", jobs: ["a4-b8-upper-control-arms"] },
      { text: "A sticking brake caliper (one wheel much hotter after a drive)", jobs: ["a4-b8-front-brakes", "a4-b8-rear-brakes"] },
    ],
  },
  {
    id: "tpms", area: "Wheels and tires",
    symptom: "Tire pressure warning light",
    first: ["Look at all four tires; check the pressures when cold"],
    causes: [
      { text: "Puncture or slow leak", jobs: ["a4-b8-flat-tire"] },
      { text: "Pressures changed or wheels swapped without resetting the monitor", jobs: ["a4-b8-wheel-swap"] },
    ],
  },
  {
    id: "shifting", area: "Transmission",
    symptom: "Jerky shifts, slipping, or shudder when pulling away (automatic)",
    first: ["Read the transmission fault codes", "Check for fluid leaks under the transmission"],
    causes: [{ text: "Old or low transmission fluid; adaptation needed after a service", jobs: ["a4-b8-transmission-fluid"] }],
  },
  {
    id: "fog", area: "Comfort and visibility",
    symptom: "Windows fog up, musty smell from the vents, or weak airflow",
    first: ["Check whether the air conditioning is on and working"],
    causes: [{ text: "Clogged cabin filter", jobs: ["a4-b8-cabin-filter"] }],
  },
  {
    id: "wipers", area: "Comfort and visibility",
    symptom: "Wipers streak, judder or squeak",
    first: ["Clean the windshield and the blades, then try again"],
    causes: [{ text: "Worn wiper blades", jobs: ["a4-b8-wipers"] }],
  },
  {
    id: "headlight", area: "Comfort and visibility",
    symptom: "Headlight out or dim",
    first: ["Check both low and high beam, and the fuse if both sides are out"],
    causes: [{ text: "Failed bulb (halogen lamps)", jobs: ["a4-b8-bulbs"] }],
  },
];
