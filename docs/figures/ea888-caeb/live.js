const D = __DATA__;
const M = makeModel();
const MOV = makeMoving({ iso, solidSvg, lineSvg, dotsSvg, pathOf, slabOf, cylinder, ring, sideArc, sideRing, segment, planOutline, lineOnTop, insetPlan }, G, D.P, M);
const root = document.querySelector(".eng");
const stage = root.querySelector(".eng-stage");
const engine = stage.querySelector(".eng-engine");
const readout = root.querySelector("[data-readout]");
const stepButtons = [...root.querySelectorAll(".eng-step")];
const startButton = root.querySelector("[data-start]");
const startLabel = root.querySelector("[data-start-label]");
const note = root.querySelector("[data-note]");
const heatEls = [...engine.querySelectorAll(".eng-heat, .eng-heat-fill")].map((el) => ({ el, cyl: Number(el.dataset.cyl), core: el.classList.contains("eng-heat-core"), fill: el.classList.contains("eng-heat-fill") }));
const LIGHT = document.body.dataset.theme === "light";
// Hot-metal inks by temperature (°C): dull red to orange, never toward pink.
const STOPS = [[560, [0.45, 0.09, 0.05]], [680, [0.56, 0.07, 0.04]], [780, [0.78, 0.1, 0.03]], [950, [0.95, 0.38, 0.06]], [1150, [1, 0.78, 0.12]]];
const inkOf = (t) => {
  if (t <= STOPS[0][0]) return STOPS[0][1];
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
    const [t0, a] = STOPS[i - 1], [t1, b] = STOPS[i];
    const u = (t - t0) / (t1 - t0);
    return a.map((v, j) => v + (b[j] - v) * u);
  }
  return STOPS[STOPS.length - 1][1];
};
const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const RUN_SPIN = 360; // shown speed while running: 60 rpm, an idle slowed about 12 times
const pulse = [0, 0, 0, 0, 0];
const warm = { x: 0, v: 0 };
let running = false;
const still = matchMedia("(prefers-reduced-motion: reduce)");

const nodes = [...engine.querySelectorAll(".it")].map((el) => ({
  el,
  piece: el.dataset.piece,
  key: Number(el.dataset.k),
  m: el.dataset.m.split(",").map(Number),
  live: el.dataset.live || "",
  shown: "",
}));
for (const node of nodes) node.lift = node.m[0] * D.V[0] + node.m[1] * D.V[1] + node.m[2] * D.V[2];
let order = nodes.map((_, index) => index);
const runs = [...engine.querySelectorAll(".eng-run")].map((el) => ({ el, r: Number(el.dataset.r), o: Number(el.dataset.o || 0) }));

const IDS = Object.keys(D.pieces);
const N = D.build.length;
const piece = Object.fromEntries(IDS.map((id) => [id, { x: D.build.includes(id) ? 1 : 0, v: 0 }]));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

let step = 0;
let theta = 0;
let spin = 120; // crank degrees per second while it turns by itself
let turning = !still.matches;
let touring = !still.matches;
let tourClock = 0;
let idleTimer = 0;
let visible = false;
let frame = 0;
let last = 0;
let drawnTheta = NaN;
let dirty = true;

const targetOf = (id) => {
  const index = D.build.indexOf(id);
  return index < 0 ? 0 : index < step ? 0 : 1;
};

function springs(dt) {
  let moving = false;
  const { stiffness, damping, substeps, stagger } = D.spring;
  D.build.forEach((id, index) => {
    const p = piece[id];
    const target = targetOf(id);
    // Parts going on wait their turn in build order; parts coming off go in reverse.
    const wait = target === 0 ? index * stagger : (N - 1 - index) * stagger;
    p.hold = p.target === target ? p.hold : wait;
    p.target = target;
    if (still.matches) { p.x = target; p.v = 0; p.hold = 0; return; }
    if (p.hold > 0) { p.hold -= dt; moving = true; return; }
    const h = dt / substeps;
    for (let i = 0; i < substeps; i++) {
      p.v += ((target - p.x) * stiffness - p.v * damping) * h;
      p.x += p.v * h;
    }
    if (Math.abs(target - p.x) < 1e-3 && Math.abs(p.v) < 1e-3) { p.x = target; p.v = 0; } else moving = true;
  });
  return moving;
}

const offOf = (id) => D.pieces[id].move.map((value) => value * piece[id].x);

// Live parts keep their children; each frame only parts whose look changed are redrawn.
for (const node of nodes) {
  if (!node.live) continue;
  node.parts = new Map([...node.el.querySelectorAll(":scope > g[data-p]")].map((el) => [el.dataset.p, { el, sig: "" }]));
  for (const part of MOV[node.live](0)) if (node.parts.has(part.id)) node.parts.get(part.id).sig = part.sig;
}
// Parts sealed inside the engine stop redrawing once these assemblies are on.
const SEALED = { crank: ["pan", "tcover"], crankFront: ["tcover"], pistons: ["head"], balance: ["tcover"], balanceFront: ["tcover"], cams: ["cover", "tcover"], camsFront: ["tcover"], valves: ["cover"] };
const sealed = (live) => (SEALED[live] || []).every((id) => piece[id] && piece[id].x === 0 && D.build.indexOf(id) < step) && piece[nodesPiece(live)].x === 0;
const nodesPiece = (live) => ({ crank: "crank", crankFront: "crank", pistons: "pistons", balance: "balance", balanceFront: "balance", cams: "cams", camsFront: "cams", valves: "head" })[live];

function draw() {
  for (const node of nodes) {
    const share = piece[node.piece].x;
    const t = share ? translateAlong(D.P, node.m.map((value) => value * share)) : "";
    if (node.shown !== t) { node.el.setAttribute("transform", t); node.shown = t; }
  }
  const angleChanged = theta !== drawnTheta;
  if (angleChanged) for (const node of nodes) {
    if (!node.live || sealed(node.live)) continue;
    for (const part of MOV[node.live](theta)) {
      const slot = node.parts.get(part.id);
      if (!slot || slot.sig === part.sig) continue;
      slot.el.innerHTML = part.svg();
      slot.sig = part.sig;
    }
  }
  if (angleChanged) {
    // Chains and belt: a run moves the drive sprocket's circumference per radian.
    const k = D.scale * theta * Math.PI / 180;
    for (const run of runs) run.el.style.strokeDashoffset = (run.o - k * run.r).toFixed(2);
  }
  drawnTheta = theta;
  const next = nodes.map((_, index) => index).sort((a, b) => nodes[a].key + nodes[a].lift * piece[nodes[a].piece].x - (nodes[b].key + nodes[b].lift * piece[nodes[b].piece].x));
  if (next.some((value, index) => value !== order[index])) {
    for (const index of next) engine.appendChild(nodes[index].el);
    order = next;
  }
  const lastIn = D.build[step - 1];
  const angle = ((theta % 720) + 720) % 720;
  const parts = [step === 0 ? "parts laid out" : `step ${step}/${N} · ${D.pieces[lastIn].say}`];
  if (step >= D.build.indexOf("crank") + 1) parts.push(`crank ${Math.round(angle % 360)}° · cyl 1 ${M.strokeOf(angle, 1)}`);
  const text = parts.join(" · ");
  if (readout.textContent !== text) readout.textContent = text;
  stepButtons.forEach((button, index) => button.toggleAttribute("data-on", index < step));
  startButton.disabled = step !== N;
  const say = step !== N ? "Finish the build to start the engine" : running ? `running · shown at 60 rpm · exhaust pulses 1-3-4-2` : warm.x > 0.05 ? "cooling down" : "Ready: start the engine";
  if (note.textContent !== say) note.textContent = say;
}

function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) {
    tourClock += dt;
    const cycle = N * 1.8 + 11;
    const t = tourClock % cycle;
    const want = t < N * 1.8 ? Math.floor(t / 1.8) + 1 : t < N * 1.8 + 9 ? N : 0;
    if (want !== step) setStep(want, false);
    const shouldRun = t > N * 1.8 + 0.8 && t < N * 1.8 + 6.5;
    if (shouldRun !== running && step === N) setRunning(shouldRun);
  }
  const moving = springs(dt);
  if (running && step < N) setRunning(false);
  if (turning || running) theta += (running ? RUN_SPIN : spin) * dt;
  const hot = heat(dt);
  draw();
  frame = visible && (moving || turning || touring || running || hot) ? requestAnimationFrame(tick) : 0;
}
const run = () => {
  if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); }
};
// Exhaust heat: the manifold warms over a few seconds of running, and each runner
// flares while its cylinder's exhaust valve is open (firing order 1-3-4-2).
function heat(dt) {
  const target = running ? 1 : 0;
  const tau = running ? 2.2 : 3.5;
  warm.x = still.matches ? target : warm.x + (target - warm.x) * (1 - Math.exp(-dt / tau));
  const angle = ((theta % 720) + 720) % 720;
  for (const c of [1, 2, 3, 4]) {
    const open = running ? M.lift(angle, c, "exhaust") / M.valveLift.exhaust : 0;
    pulse[c] = Math.max(open, pulse[c] * Math.exp(-dt / 0.18));
  }
  pulse[0] = (pulse[1] + pulse[2] + pulse[3] + pulse[4]) / 4;
  for (const h of heatEls) {
    const t = 480 + warm.x * (h.cyl ? 250 + 160 * pulse[h.cyl] : 330);
    const glow = smoothstep(560, 760, t) * warm.x;
    const ink = inkOf(h.core ? t + 60 : t);
    const css = `rgb(${ink.map((v) => Math.round(v * 255)).join(",")})`;
    const alpha = (LIGHT ? 0.78 : 1) * glow * (h.fill ? 0.55 : h.core ? 0.9 : 0.75);
    if (h.fill) h.el.style.fill = css; else h.el.style.stroke = css;
    h.el.style.opacity = alpha.toFixed(3);
  }
  return warm.x > 0.002 && !running ? true : running;
}
function setRunning(on) {
  running = on && step === N;
  startLabel.textContent = running ? "Stop" : "Start";
  startButton.toggleAttribute("data-on", running);
  run();
}
function setStep(n, user = true) {
  step = clamp(n, 0, N);
  stage.setAttribute("aria-valuenow", String(step));
  stage.setAttribute("aria-valuetext", step === 0 ? "All assemblies laid out" : `Step ${step}: ${D.pieces[D.build[step - 1]].name}`);
  if (user) pause();
  run();
}
function pause() {
  touring = false;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => { touring = !still.matches; tourClock = step * 1.8; run(); }, 6000);
}

new IntersectionObserver((entries) => {
  visible = entries[entries.length - 1].isIntersecting;
  if (visible) run();
}, { rootMargin: "120px 0px" }).observe(stage);

let dragFrom = null;
stage.addEventListener("pointermove", (event) => {
  if (dragFrom) {
    theta = dragFrom.theta + (event.clientX - dragFrom.x) * 2;
    pause();
    run();
    return;
  }
  if (event.pointerType !== "mouse") return;
  const box = stage.getBoundingClientRect();
  const share = clamp((event.clientX - box.left) / box.width, 0, 0.999);
  const n = Math.round(share * N * 1.15);
  if (n !== step) setStep(Math.min(N, n));
});
stage.addEventListener("pointerdown", (event) => {
  dragFrom = { x: event.clientX, theta };
  turning = false;
  stage.setPointerCapture(event.pointerId);
});
const release = () => { dragFrom = null; };
stage.addEventListener("pointerup", release);
stage.addEventListener("pointercancel", release);
stage.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight" || event.key === "ArrowUp") setStep(step + 1);
  else if (event.key === "ArrowLeft" || event.key === "ArrowDown") setStep(step - 1);
  else if (event.key === "Home") setStep(0);
  else if (event.key === "End") setStep(N);
  else if (event.key === " ") { turning = !turning; pause(); run(); }
  else if (event.key === "s" || event.key === "S" || event.key === "Enter") { setRunning(!running); pause(); }
  else return;
  event.preventDefault();
});
stepButtons.forEach((button) => button.addEventListener("click", () => setStep(Number(button.dataset.step))));
startButton.addEventListener("click", () => { setRunning(!running); pause(); });
draw();
