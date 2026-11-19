// EA888 rebuild trainer: the step engine. Data (steps, specs, diagrams, figures) is
// inlined by build.mjs as `DATA`.
const D = __DATA__;
const STORE = "ea888-rebuild-v1";
const root = document.querySelector(".rb");
const figBox = root.querySelector("[data-figure]");
const card = root.querySelector("[data-card]");
const list = root.querySelector("[data-list]");
const bar = root.querySelector("[data-bar]");
const readout = root.querySelector("[data-readout]");
const modeButtons = [...root.querySelectorAll("[data-mode]")];
const still = matchMedia("(prefers-reduced-motion: reduce)");

// ---------------------------------------------------------------- state (saved in this browser)
const blank = () => ({ step: 0, mode: "game", variants: {}, done: {}, answers: {}, checks: {}, values: {}, order: {}, torque: {}, mistakes: 0 });
let S = blank();
try {
  const saved = JSON.parse(localStorage.getItem(STORE) || "null");
  if (saved && typeof saved === "object") S = { ...blank(), ...saved };
} catch (error) {
  /* storage blocked: progress lives for this visit only */
}
const save = () => {
  try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (error) { /* ignore */ }
};

// ---------------------------------------------------------------- variants and visibility
const matches = (when) => !when || Object.entries(when).every(([key, want]) => [want].flat().includes(S.variants[key]));
const steps = () => D.steps.filter((step) => matches(step.when));
const tasksOf = (step) => step.tasks.map((task, index) => ({ ...task, index })).filter((task) => matches(task.when));
const key = (step, task, extra = "") => `${step.id}:${task.index}${extra}`;

function specFor(id) {
  if (D.spec[id]) return D.spec[id];
  const two = S.variants.oilRing === "2";
  const table = {
    gap1: two ? { min: 0.2, max: 0.4, limit: 0.8 } : { min: 0.3, max: 0.4, limit: 0.6 },
    gap2: two ? { min: 0.2, max: 0.4, limit: 0.8 } : { min: 0.4, max: 0.5, limit: 0.7 },
    gapOil: { min: 0.25, max: 0.5, limit: 0.8 },
    groove1: { min: 0.06, max: 0.09, limit: 0.2 },
    groove2: { min: 0.03, max: 0.06, limit: 0.15 },
  }[id];
  return { ...table, unit: "mm", ref: id.startsWith("gap") ? "13 · p. 57" : "13 · p. 58" };
}
const fmt = (value) => (Math.abs(value) < 1 ? value.toFixed(3).replace(/0$/, "") : value.toFixed(2));
const specText = (spec) => (spec.nominal ? `${fmt(spec.nominal)} mm ± ${fmt(spec.dev)} mm` : spec.limit === undefined ? (spec.min === 0 ? `max ${fmt(spec.max)} mm` : `${fmt(spec.min)}–${fmt(spec.max)} mm`) : `new ${fmt(spec.min)}–${fmt(spec.max)} mm · wear limit ${fmt(spec.limit)} mm`);
function grade(spec, value) {
  if (!Number.isFinite(value)) return null;
  if (spec.nominal) return Math.abs(value - spec.nominal) <= spec.dev + 1e-9 ? "ok" : "bad";
  if (value >= spec.min - 1e-9 && value <= spec.max + 1e-9) return "ok";
  if (value < spec.min) return "low";
  return value <= spec.limit + 1e-9 ? "worn" : "bad";
}
const GRADE = { ok: "in spec", worn: "past new spec, within wear limit", low: "below new spec", bad: "out of limit: replace or repair" };

// ---------------------------------------------------------------- task completion
function taskDone(step, task) {
  const k = key(step, task);
  if (task.type === "note") return true;
  if (task.type === "variant") return task.keys.every((name) => S.variants[name]);
  if (task.type === "check") return task.items.every((_, i) => S.checks[`${k}.${i}`]);
  if (task.type === "choice") return S.answers[k] === "ok";
  if (task.type === "order") return (S.order[k] || []).length === D.diagrams[task.diagram].bolts.length;
  if (task.type === "torque") return task.stages.every((_, i) => S.torque[`${k}.${i}`]);
  if (task.type === "measure") return task.items.every((_, i) => Number.isFinite(S.values[`${k}.${i}`]));
  return true;
}
const stepDone = (step) => tasksOf(step).every((task) => taskDone(step, task));
function markDone(step) {
  const was = !!S.done[step.id];
  S.done[step.id] = stepDone(step);
  return S.done[step.id] && !was;
}

// ---------------------------------------------------------------- figure
const figures = {};
for (const [id, svg] of Object.entries(D.figures)) figures[id] = svg;
let shownFig = "";
let lastInstalled = new Set();
function renderFigure(step, fresh) {
  const all = steps();
  const index = all.indexOf(step);
  const before = new Set(all.slice(0, index).flatMap((s) => s.install || []));
  const later = new Set(all.slice(index + 1).flatMap((s) => s.install || []));
  const mine = new Set(step.install || []);
  const done = stepDone(step);
  if (shownFig !== step.figure) {
    figBox.innerHTML = figures[step.figure];
    const svg = figBox.querySelector("svg");
    svg.setAttribute("data-view", svg.getAttribute("viewBox"));
    shownFig = step.figure;
    lastInstalled = new Set();
  }
  for (const g of figBox.querySelectorAll("[data-part]")) {
    const name = g.dataset.part;
    const [dx, dy] = g.dataset.apart.split(",").map(Number);
    let state = "placed";
    if (mine.has(name)) state = done ? "placed" : "apart";
    else if (before.has(name)) state = "placed";
    else if (later.has(name)) state = "hidden";
    g.style.display = state === "hidden" ? "none" : "";
    // Slide home when the step completes; jump when the figure or step changes.
    const target = state === "apart" ? `translate(${dx}px, ${dy}px)` : "translate(0px, 0px)";
    if (g.style.transform !== target) {
      g.style.transition = fresh || still.matches ? "none" : "transform .9s cubic-bezier(.32,.72,0,1)";
      g.style.transform = target;
    }
    const lit = (step.focus || []).includes(name);
    g.toggleAttribute("data-focus", lit);
    for (const solid of g.querySelectorAll(".iso-solid")) solid.toggleAttribute("data-lit", lit);
  }
  // Once the step is done, zoom to the assembled parts after they have slid home.
  clearTimeout(frameTimer);
  if (done && !fresh && !still.matches) frameTimer = setTimeout(() => frameFigure(new Set()), 950);
  else frameFigure(done ? new Set() : mine);
}
let frameTimer = 0;

// Zoom the figure to what it shows in this step: every visible part where it sits, plus the
// parts of this step where they wait, so nothing jumps as they slide home.
function frameFigure(mine) {
  const svg = figBox.querySelector("svg");
  if (!svg) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const g of svg.querySelectorAll("[data-part]")) {
    if (g.style.display === "none") continue;
    const box = g.getBBox();
    if (!box.width && !box.height) continue;
    const [dx, dy] = g.dataset.apart.split(",").map(Number);
    const shifts = mine.has(g.dataset.part) ? [[0, 0], [dx, dy]] : [[0, 0]];
    for (const [sx, sy] of shifts) {
      x0 = Math.min(x0, box.x + sx); y0 = Math.min(y0, box.y + sy);
      x1 = Math.max(x1, box.x + box.width + sx); y1 = Math.max(y1, box.y + box.height + sy);
    }
  }
  if (!Number.isFinite(x0)) return;
  const [, , W, H] = svg.getAttribute("data-view").split(" ").map(Number);
  // Keep the figure's aspect ratio and at least 45% of its full size, so small parts are not blown up.
  let w = Math.max(x1 - x0 + 40, W * 0.45), h = Math.max(y1 - y0 + 40, H * 0.45);
  if (w / h > W / H) h = (w * H) / W; else w = (h * W) / H;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  svg.style.transition = "none";
  svg.setAttribute("viewBox", `${(cx - w / 2).toFixed(1)} ${(cy - h / 2).toFixed(1)} ${w.toFixed(1)} ${h.toFixed(1)}`);
}

// ---------------------------------------------------------------- task views
const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const garage = () => S.mode === "garage";

function variantView(step, task) {
  return task.keys.map((name) => {
    const v = D.variants[name];
    return `<fieldset class="rb-variant"><legend>${esc(v.q)}</legend><p class="rb-help">${esc(v.help)}</p><div class="rb-options">${v.options.map((o) => `<button type="button" class="rb-opt" data-variant="${name}" data-value="${o.id}" ${S.variants[name] === o.id ? "data-on" : ""}>${esc(o.label)}</button>`).join("")}</div></fieldset>`;
  }).join("");
}
function checkView(step, task) {
  const k = key(step, task);
  return `<ul class="rb-checks">${task.items.map((item, i) => {
    const [text, ref] = item.split(" · ");
    return `<li><label><input type="checkbox" data-check="${k}.${i}" ${S.checks[`${k}.${i}`] ? "checked" : ""}><span>${esc(text)}${ref ? `<em>${esc(ref)}</em>` : ""}</span></label></li>`;
  }).join("")}</ul>`;
}
function choiceView(step, task) {
  const k = key(step, task);
  const answer = S.answers[k];
  if (garage()) {
    const right = task.options.find((o) => o.ok);
    return `<div class="rb-q"><p>${esc(task.q)}</p><p class="rb-answer"><strong>${esc(right.text)}.</strong> ${esc(right.why)}</p></div>`;
  }
  const picked = S.answers[`${k}#picked`];
  return `<div class="rb-q"><p>${esc(task.q)}</p><div class="rb-options rb-col">${task.options.map((o, i) => {
    const state = picked === i ? (o.ok ? "right" : "wrong") : answer === "ok" && o.ok ? "right" : "";
    return `<button type="button" class="rb-opt" data-choice="${k}" data-i="${i}" ${state ? `data-state="${state}"` : ""} ${answer === "ok" ? "disabled" : ""}>${esc(o.text)}</button>`;
  }).join("")}</div>${picked !== undefined ? `<p class="rb-why" data-state="${task.options[picked].ok ? "right" : "wrong"}">${esc(task.options[picked].why)}</p>` : ""}</div>`;
}
function orderView(step, task) {
  const k = key(step, task);
  const d = D.diagrams[task.diagram];
  const got = S.order[k] || [];
  const next = got.length + 1;
  const total = d.bolts.length;
  let shapes = "";
  let marks = "";
  const W = 300, H = 300;
  if (task.diagram === "mains") {
    // Five caps in a column, as in the manual's figure; side bolt arrows at caps 2–4.
    const capH = 34, gap = 18, top = 22;
    for (let c = 0; c < d.caps; c++) {
      const y = top + c * (capH + gap);
      shapes += `<rect class="rb-cap" x="70" y="${y}" width="160" height="${capH}" rx="5"/>`;
      if (d.side.includes(c)) shapes += `<path class="rb-side" d="M18 ${y + capH / 2}h34m-8 -6l8 6l-8 6M282 ${y + capH / 2}h-34m8 -6l-8 6l8 6"/><text class="rb-side-t" x="10" y="${y + capH / 2 + 4}">A</text><text class="rb-side-t" x="284" y="${y + capH / 2 + 4}">A</text>`;
    }
    d.bolts.forEach((b, i) => {
      const x = b.side ? 205 : 95;
      const y = top + b.cap * (capH + gap) + capH / 2;
      marks += boltMark(k, b, i, x, y, got);
    });
    shapes += `<text class="rb-end" x="150" y="${top - 8}">${esc("bearing cap 1 / 5: check orientation in your manual, p. 38")}</text>`;
  } else {
    // Outlines roughly as in the manual's figures: rear flange, timing covers, bearing bracket, oil pans.
    const outline = {
      flange: `<path class="rb-cap" d="M60 40 Q150 -6 240 40 L262 210 Q150 300 38 210 Z"/><circle class="rb-hub" cx="150" cy="150" r="52"/>`,
      cover: `<path class="rb-cap" d="M70 26 Q150 4 230 26 L262 120 L244 230 Q150 296 56 230 L38 120 Z"/><circle class="rb-hub" cx="150" cy="190" r="30"/>`,
      bracket: `<path class="rb-cap" d="M40 92 L150 34 L262 124 L196 260 L72 256 Z"/><circle class="rb-hub" cx="186" cy="132" r="34"/>`,
      pan: `<rect class="rb-cap" x="28" y="30" width="244" height="232" rx="14"/>`,
    }[d.shape || "flange"];
    shapes += outline;
    d.bolts.forEach((b, i) => { marks += boltMark(k, b, i, 30 + b.x * 2.4, 22 + b.y * 2.4, got); });
  }
  const status = got.length === total ? `<p class="rb-why" data-state="right">Right: ${esc(d.ref)}. ${task.diagram === "mains" ? "Middle cap first, then outward." : ""}</p>` : `<p class="rb-help">${garage() ? "Order shown as in the manual." : `Click bolt ${next} of ${total}.`}</p>`;
  return `<div class="rb-q"><p>${esc(task.q)}</p><svg class="rb-diagram" viewBox="0 0 ${W} ${H}" role="group" aria-label="Bolt pattern">${shapes}${marks}</svg>${status}</div>`;
}
function boltMark(k, b, i, x, y, got) {
  const known = garage() || got.includes(b.n);
  return `<g class="rb-bolt" data-order="${k}" data-n="${b.n}" ${known ? "data-on" : ""} tabindex="0" role="button" aria-label="${known ? `Bolt ${b.n}` : "Bolt"}"><circle cx="${x}" cy="${y}" r="11"/>${known ? `<text x="${x}" y="${y + 4}">${b.n}</text>` : ""}</g>`;
}
function torqueView(step, task) {
  const k = key(step, task);
  return `<div class="rb-q"><p><strong>${esc(task.what)}</strong>${task.tool ? ` · ${esc(task.tool)}` : ""}</p>${task.note ? `<p class="rb-help">${esc(task.note)}</p>` : ""}<ol class="rb-stages">${task.stages.map((st, i) => {
    const sk = `${k}.${i}`;
    const ok = S.torque[sk];
    const value = st.nm !== undefined ? `${st.nm} Nm` : st.deg !== undefined ? `${st.deg}° further` : "";
    if (st.check) return `<li><label><input type="checkbox" data-torque="${sk}" ${ok ? "checked" : ""}><span>${esc(st.bolts)}</span></label></li>`;
    if (garage() || ok) return `<li ${ok ? "data-on" : ""}><span>${esc(st.bolts)}</span><b>${value}</b>${garage() ? `<label class="rb-tick"><input type="checkbox" data-torque="${sk}" ${ok ? "checked" : ""}> done</label>` : ""}</li>`;
    const unit = st.nm !== undefined ? "Nm" : "°";
    return `<li><span>${esc(st.bolts)}</span><span class="rb-entry"><input type="number" inputmode="decimal" data-torque-in="${sk}" aria-label="${esc(st.bolts)} in ${unit}"><em>${unit}</em><button type="button" class="rb-opt" data-torque-set="${sk}">Set</button></span><span class="rb-hint" data-hint="${sk}"></span></li>`;
  }).join("")}</ol></div>`;
}
function measureView(step, task) {
  const k = key(step, task);
  const spec = specFor(task.spec);
  const rows = task.items.map((item, i) => {
    const value = S.values[`${k}.${i}`];
    const g = grade(spec, value);
    return `<tr><th>${esc(item)}</th><td><input type="number" step="0.001" inputmode="decimal" data-value="${k}.${i}" value="${Number.isFinite(value) ? value : ""}" aria-label="${esc(item)} in mm"></td><td>${g ? `<span class="rb-grade" data-grade="${g}">${GRADE[g]}</span>` : ""}</td></tr>`;
  }).join("");
  const bad = task.items.some((_, i) => ["bad", "low"].includes(grade(spec, S.values[`${k}.${i}`])));
  return `<div class="rb-q"><p><strong>${esc(task.label)}</strong> · ${specText(spec)} <em>${esc(spec.ref)}</em></p><div class="rb-table-wrap"><table class="rb-table">${rows}</table></div><div class="rb-row"><button type="button" class="rb-opt" data-practice="${k}">Fill practice readings</button><button type="button" class="rb-opt" data-clear="${k}">Clear</button></div>${bad ? `<p class="rb-why" data-state="wrong">At least one reading is outside the manual's spec. Do not build on it: find the cause and replace or repair the part first.</p>` : ""}</div>`;
}
const VIEWS = { variant: variantView, check: checkView, choice: choiceView, order: orderView, torque: torqueView, measure: measureView, note: (step, task) => `<p class="rb-note">${esc(task.text)}</p>` };

// ---------------------------------------------------------------- render
function render(fresh = false) {
  const all = steps();
  S.step = Math.max(0, Math.min(S.step, all.length - 1));
  const step = all[S.step];
  const group = D.groups.find((g) => g.id === step.group);
  const done = stepDone(step);
  const tasks = tasksOf(step);
  card.innerHTML = `
    <p class="rb-kicker">${esc(group.name)} · step ${S.step + 1} of ${all.length}</p>
    <h2>${esc(step.title)}</h2>
    <p class="rb-say">${esc(step.say)}</p>
    <p class="rb-ref">Repair manual, Rep. Gr. ${esc(step.ref)}</p>
    ${tasks.map((task) => `<section class="rb-task" ${taskDone(step, task) ? "data-done" : ""}>${VIEWS[task.type](step, task)}</section>`).join("")}
    <div class="rb-nav"><button type="button" class="rb-btn" data-go="-1" ${S.step === 0 ? "disabled" : ""}>Back</button><span class="rb-state">${done ? "Step complete" : garage() ? "Tick off as you go" : "Answer every task to continue"}</span><button type="button" class="rb-btn rb-primary" data-go="1" ${S.step === all.length - 1 || (!done && !garage()) ? "disabled" : ""}>Next step</button></div>`;
  renderFigure(step, fresh);
  // Step list, by group.
  list.innerHTML = D.groups.map((g) => {
    const items = all.map((s, i) => [s, i]).filter(([s]) => s.group === g.id);
    if (!items.length) return "";
    return `<div class="rb-group"><h3>${esc(g.name)}</h3><ol>${items.map(([s, i]) => `<li><button type="button" class="rb-step" data-jump="${i}" ${i === S.step ? "aria-current=\"step\"" : ""} ${S.done[s.id] ? "data-done" : ""}><span class="rb-dot" aria-hidden="true"></span>${esc(s.title)}</button></li>`).join("")}</ol></div>`;
  }).join("");
  const count = all.filter((s) => S.done[s.id]).length;
  bar.style.setProperty("--p", `${(count / all.length) * 100}%`);
  readout.textContent = `${count}/${all.length} steps done${S.mistakes ? ` · ${S.mistakes} wrong answer${S.mistakes === 1 ? "" : "s"}` : ""}`;
  modeButtons.forEach((b) => b.toggleAttribute("data-on", b.dataset.mode === S.mode));
  root.dataset.mode = S.mode;
}

function update(step, slide = true) {
  const newly = markDone(step);
  save();
  render(!slide);
  if (newly) card.querySelector(".rb-state")?.classList.add("rb-pop");
}
const current = () => steps()[S.step];

// ---------------------------------------------------------------- interaction
root.addEventListener("click", (event) => {
  const t = event.target.closest("button, [data-order]");
  if (!t) return;
  const step = current();
  if (t.dataset.variant) {
    S.variants[t.dataset.variant] = t.dataset.value;
    return update(step);
  }
  if (t.dataset.choice) {
    const task = tasksOf(step).find((x) => key(step, x) === t.dataset.choice);
    const i = Number(t.dataset.i);
    S.answers[`${t.dataset.choice}#picked`] = i;
    if (task.options[i].ok) S.answers[t.dataset.choice] = "ok";
    else S.mistakes++;
    return update(step);
  }
  if (t.dataset.order) {
    if (garage()) return;
    const k = t.dataset.order;
    const got = S.order[k] || (S.order[k] = []);
    const n = Number(t.dataset.n);
    if (got.includes(n)) return;
    if (n === got.length + 1) {
      got.push(n);
      return update(step);
    }
    S.mistakes++;
    t.classList.remove("rb-shake");
    void t.getBoundingClientRect();
    t.classList.add("rb-shake");
    readout.textContent = `Not that one: bolt ${got.length + 1} comes next. ${current().id === "main-torque" ? "Work from the middle cap outward." : ""}`;
    save();
    return;
  }
  if (t.dataset.torqueSet) {
    const sk = t.dataset.torqueSet;
    const [stepId, rest] = sk.split(":");
    const [ti, si] = rest.split(".").map(Number);
    const st = step.tasks[ti].stages[si];
    const input = card.querySelector(`[data-torque-in="${sk}"]`);
    const value = Number(input.value);
    const want = st.nm !== undefined ? st.nm : st.deg;
    const hint = card.querySelector(`[data-hint="${sk}"]`);
    if (value === want) {
      S.torque[sk] = true;
      return update(step);
    }
    S.mistakes++;
    hint.textContent = input.value === "" ? "Enter a value." : value > want ? "Too high: check the manual's figure." : "Too low: check the manual's figure.";
    save();
    void stepId;
    return;
  }
  if (t.dataset.practice) {
    const k = t.dataset.practice;
    const task = tasksOf(step).find((x) => key(step, x) === k);
    const spec = specFor(task.spec);
    // Practice readings: mostly good parts, sometimes one that is worn or out of limit.
    task.items.forEach((_, i) => {
      const roll = Math.random();
      let v;
      if (spec.nominal) v = spec.nominal + (roll < 0.9 ? (Math.random() * 2 - 1) * spec.dev * 0.7 : spec.dev * (1.2 + Math.random() * 0.5));
      else if (spec.limit === undefined) v = roll < 0.85 ? spec.min + Math.random() * (spec.max - spec.min) : spec.max + (spec.max - spec.min) * (0.5 + Math.random());
      else v = roll < 0.8 ? spec.min + Math.random() * (spec.max - spec.min) : roll < 0.95 ? spec.max + Math.random() * (spec.limit - spec.max) : spec.limit * (1.05 + Math.random() * 0.2);
      S.values[`${k}.${i}`] = Number(v.toFixed(3));
    });
    return update(step);
  }
  if (t.dataset.clear) {
    for (const name of Object.keys(S.values)) if (name.startsWith(`${t.dataset.clear}.`)) delete S.values[name];
    return update(step);
  }
  if (t.dataset.go) {
    S.step += Number(t.dataset.go);
    save();
    render(true);
    root.querySelector(".rb-main")?.scrollIntoView({ block: "start", behavior: still.matches ? "auto" : "smooth" });
    return;
  }
  if (t.dataset.jump) {
    S.step = Number(t.dataset.jump);
    save();
    return render(true);
  }
  if (t.dataset.mode) {
    S.mode = t.dataset.mode;
    save();
    return render(true);
  }
  if (t.dataset.action === "export") return exportLog();
  if (t.dataset.action === "print") return window.print();
  if (t.dataset.action === "reset") {
    if (confirm("Clear all progress, answers and measurements saved in this browser?")) {
      S = blank();
      save();
      render(true);
    }
  }
});
root.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-order]")) {
    event.preventDefault();
    event.target.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  }
  if (event.key === "Enter" && event.target.matches("[data-torque-in]")) card.querySelector(`[data-torque-set="${event.target.dataset.torqueIn}"]`)?.click();
});
root.addEventListener("change", (event) => {
  const t = event.target;
  const step = current();
  if (t.dataset.check) { S.checks[t.dataset.check] = t.checked; return update(step); }
  if (t.dataset.torque) { S.torque[t.dataset.torque] = t.checked; return update(step); }
  if (t.dataset.value) {
    const value = t.value === "" ? NaN : Number(t.value);
    if (Number.isFinite(value)) S.values[t.dataset.value] = value; else delete S.values[t.dataset.value];
    return update(step);
  }
});

// A log of everything entered, to keep with the engine's paperwork.
function exportLog() {
  const all = steps();
  const out = { engine: "EA888 Gen 2 · CAEB", exported: new Date().toISOString(), variants: S.variants, steps: [] };
  for (const step of all) {
    const entry = { id: step.id, title: step.title, ref: step.ref, done: !!S.done[step.id], measurements: [] };
    for (const task of tasksOf(step)) {
      if (task.type !== "measure") continue;
      const spec = specFor(task.spec);
      task.items.forEach((item, i) => {
        const value = S.values[`${key(step, task)}.${i}`];
        if (Number.isFinite(value)) entry.measurements.push({ what: `${task.label}: ${item}`, value, unit: "mm", spec: specText(spec), result: GRADE[grade(spec, value)] });
      });
    }
    out.steps.push(entry);
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ea888-rebuild-log.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// Printable checklist: every step with its specs, built once.
const sheet = root.querySelector("[data-print]");
function printSheet() {
  const all = steps();
  sheet.innerHTML = `<h1>EA888 Gen 2 (CAEB) · assembly checklist (stages 1–3)</h1><p>Versions: rod bolts ${esc(S.variants.rodBolt || "?")}, oil ring ${esc(S.variants.oilRing ? `${S.variants.oilRing}-piece` : "?")}, flange ${esc(S.variants.flange || "?")}. Values from the Audi repair manual for EA888 Gen II (06.2019); check each against your copy.</p>` + all.map((step, i) => {
    const lines = tasksOf(step).map((task) => {
      if (task.type === "check") return task.items.map((x) => `<li>☐ ${esc(x)}</li>`).join("");
      if (task.type === "choice") { const r = task.options.find((o) => o.ok); return `<li>☐ ${esc(task.q)} <b>${esc(r.text)}</b></li>`; }
      if (task.type === "torque") return `<li>☐ ${esc(task.what)}: ${task.stages.map((st) => (st.check ? esc(st.bolts) : `${esc(st.bolts)} ${st.nm !== undefined ? `${st.nm} Nm` : `${st.deg}°`}`)).join(" → ")}</li>`;
      if (task.type === "order") return `<li>☐ Tightening sequence per manual figure (Rep. Gr. ${esc(D.diagrams[task.diagram].ref)})</li>`;
      if (task.type === "measure") { const spec = specFor(task.spec); return `<li>☐ ${esc(task.label)} (${specText(spec)}): ${task.items.map((it, j) => `${esc(it)} ${Number.isFinite(S.values[`${key(step, task)}.${j}`]) ? S.values[`${key(step, task)}.${j}`] : "____"}`).join(" · ")}</li>`; }
      if (task.type === "variant") return task.keys.map((name) => `<li>☐ ${esc(D.variants[name].q)}: ${esc(S.variants[name] || "____")}</li>`).join("");
      if (task.type === "note") return `<li>${esc(task.text)}</li>`;
      return "";
    }).join("");
    return `<section><h2>${i + 1}. ${esc(step.title)} <small>Rep. Gr. ${esc(step.ref)}</small></h2><ul>${lines}</ul></section>`;
  }).join("");
}
window.addEventListener("beforeprint", printSheet);

render(true);
