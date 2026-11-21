// Builds one self-contained HTML page for a job: the job engine (runtime/game.js) with the
// job's data inlined. Also builds the catalog page that links the jobs.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../.claude/skills/anatomy/kit/iso-kit.mjs";

const RUNTIME = join(dirname(fileURLToPath(import.meta.url)), "..", "runtime");
const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// Every figure a step names must exist, and every part a step moves or lights must be drawn.
export function checkJob(meta, data) {
  const drawn = new Set(Object.values(data.figures).flatMap((svg) => [...svg.matchAll(/data-part="([^"]+)"/g)].map((m) => m[1])));
  const groups = new Set(data.groups.map((g) => g.id));
  for (const step of data.steps) {
    const where = `${meta.id} / ${step.id}`;
    if (!data.figures[step.figure]) throw new Error(`${where}: no figure "${step.figure}"`);
    if (!groups.has(step.group)) throw new Error(`${where}: no group "${step.group}"`);
    for (const name of [...(step.install || []), ...(step.remove || []), ...(step.focus || [])]) if (!drawn.has(name)) throw new Error(`${where}: no part "${name}" in any figure`);
    for (const task of step.tasks) {
      if (task.type === "order" && !data.diagrams[task.diagram]) throw new Error(`${where}: no diagram "${task.diagram}"`);
      if (task.type === "variant") for (const name of task.keys) if (!data.variants[name]) throw new Error(`${where}: no variant "${name}"`);
    }
  }
}

export function jobPage(meta, data, { theme = "light", back = "" } = {}) {
  checkJob(meta, data);
  const job = { id: meta.id, title: meta.title, vehicleLine: meta.vehicleLine, source: meta.source, storageKey: meta.storageKey, status: meta.status };
  const CSS = readFileSync(join(RUNTIME, "game.css"), "utf8");
  const GAME = readFileSync(join(RUNTIME, "game.js"), "utf8").replace("__DATA__", JSON.stringify({ job, ...data }));
  const kicker = [meta.vehicleLine, meta.status === "draft" ? "draft" : ""].filter(Boolean).join(" · ");
  const legal = meta.legal || `Practice tool. Values come from: ${meta.source.name}. Check every value against your own source before you work on the car. Progress and measurements are saved in this browser only; export them to keep a copy.`;
  const body = `<div class="rb iso-figure">
  ${back ? `<p><a class="rb-back" href="${esc(back)}">← All jobs</a></p>` : ""}
  <header class="rb-top">
    <div class="rb-title"><p class="rb-kicker">${esc(kicker)}</p><h1>${esc(meta.title)}</h1></div>
    <div class="rb-modes" role="group" aria-label="Mode"><button type="button" class="rb-mode" data-mode="game">Practice</button><button type="button" class="rb-mode" data-mode="garage">Garage</button></div>
    <div class="rb-progress"><div class="rb-bar" data-bar aria-hidden="true"></div><p class="rb-readout" data-readout aria-live="polite"></p></div>
  </header>
  ${meta.draftNote ? `<p class="rb-draft"><strong>Draft.</strong> ${esc(meta.draftNote.replace(/^Draft:\s*/, ""))}</p>` : ""}
  <div class="rb-main">
    <div class="iso-plate rb-plate"><div class="rb-figure" data-figure></div></div>
    <article class="rb-card" data-card aria-live="polite"></article>
  </div>
  <nav class="rb-list" data-list aria-label="Steps"></nav>
  <div class="rb-actions"><button type="button" class="rb-btn" data-action="export">Export measurements</button><button type="button" class="rb-btn" data-action="print">Print checklist</button><button type="button" class="rb-btn" data-action="reset">Reset progress</button></div>
  <p class="rb-legal">${esc(legal)}</p>
  <div class="rb-print" data-print></div>
</div>`;
  return k.pageHtml({ title: meta.title, theme, body: `<style>${CSS}</style>${body}`, script: GAME, width: 1180 });
}

// The catalog: jobs grouped by vehicle, with filters for system, difficulty, time, status
// and a text search. Filters live in the URL hash, so a filtered view can be bookmarked or shared.
const LEVELS = [["beginner", "Beginner"], ["intermediate", "Intermediate"], ["expert", "Expert"]];
const TIMES = [["any", "Any time"], ["30", "Up to 30 min"], ["120", "Up to 2 h"], ["long", "Longer"]];
const levelsOf = (difficulty) => LEVELS.map(([id]) => id).filter((id) => difficulty.toLowerCase().includes(id));
const carOf = (v) => `${v.make} ${v.model}${v.generation ? ` (${v.generation})` : ""}`;

export function catalogPage(jobs, { theme = "light", title = "Car fixes", symptoms = false } = {}) {
  const byCar = new Map();
  for (const job of jobs) {
    const car = carOf(job.meta.vehicle);
    if (!byCar.has(car)) byCar.set(car, []);
    byCar.get(car).push(job);
  }
  const time = (m) => (m >= 120 ? `${Math.round(m / 60)} h` : `${m} min`);
  const count = (pick) => (value) => jobs.filter((j) => pick(j).includes(value)).length;
  const systems = [...new Set(jobs.map((j) => j.meta.system))].sort();
  const cars = [...byCar.keys()].sort();

  const card = ({ meta, href, steps }) => {
    const text = [meta.title, meta.summary, meta.system, meta.engine || "", meta.vehicleLine, meta.commonProblem || ""].join(" ").toLowerCase();
    const status = meta.status === "draft" ? `<span class="cat-tag" data-draft>${esc(meta.statusLabel || "Draft · specs not loaded")}</span>` : `<span class="cat-tag">Specs loaded</span>`;
    return `<li data-job data-system="${esc(meta.system)}" data-car="${esc(carOf(meta.vehicle))}" data-levels="${levelsOf(meta.difficulty).join(" ")}" data-minutes="${meta.minutes}" data-status="${meta.status}" data-type="${meta.commonProblem ? "problem" : ""}" data-text="${esc(text)}"><a class="cat-job" href="${esc(href)}"><span class="cat-meta">${esc(meta.system)}${meta.engine ? ` · ${esc(meta.engine)}` : ""}</span>${meta.commonProblem ? `<span class="cat-problem">Common problem: ${esc(meta.commonProblem)}</span>` : ""}<strong>${esc(meta.title)}</strong><span class="cat-sum">${esc(meta.summary)}</span><span class="cat-tags">${status}<span class="cat-tag">${steps} steps</span><span class="cat-tag">about ${time(meta.minutes)}</span><span class="cat-tag">${esc(meta.difficulty)}</span>${meta.private ? `<span class="cat-tag" data-private>Private</span>` : ""}</span></a></li>`;
  };
  const sections = cars.map((car) => `<section class="cat-car" data-car-section="${esc(car)}"><h2>${esc(car)}</h2><ul class="cat-jobs">${byCar.get(car).sort((a, b) => a.meta.system.localeCompare(b.meta.system) || a.meta.title.localeCompare(b.meta.title)).map(card).join("")}</ul></section>`).join("");

  const chips = (group, items, multi = true) => `<div class="cat-chips" role="group" aria-label="${esc(group.label)}" data-group="${group.id}" data-multi="${multi ? 1 : 0}"><span class="cat-label">${esc(group.label)}</span>${items.map(([id, label, n]) => `<button type="button" class="cat-chip" data-value="${esc(id)}" aria-pressed="false">${esc(label)}${n !== undefined ? ` <span class="cat-n">${n}</span>` : ""}</button>`).join("")}</div>`;
  const filters = jobs.length ? `<div class="cat-filters">
  <label class="cat-search"><span class="cat-label">Search</span><input type="search" data-search placeholder="oil, brakes, battery…" autocomplete="off"></label>
  ${cars.length > 1 ? chips({ id: "car", label: "Car" }, cars.map((c) => [c, c, count((j) => [carOf(j.meta.vehicle)])(c)])) : ""}
  ${jobs.some((j) => j.meta.commonProblem) ? chips({ id: "type", label: "Type" }, [["problem", "Common problems", jobs.filter((j) => j.meta.commonProblem).length]]) : ""}
  ${chips({ id: "system", label: "System" }, systems.map((x) => [x, x, count((j) => [j.meta.system])(x)]))}
  ${chips({ id: "level", label: "Difficulty" }, LEVELS.map(([id, label]) => [id, label, count((j) => levelsOf(j.meta.difficulty))(id)]))}
  ${chips({ id: "time", label: "Time" }, TIMES.map(([id, label]) => [id, label]), false)}
  ${chips({ id: "status", label: "Specs" }, [["ready", "Specs loaded", count((j) => [j.meta.status])("ready")], ["draft", "Drafts", count((j) => [j.meta.status])("draft")]])}
  <p class="cat-result"><span data-result aria-live="polite"></span> <button type="button" class="cat-clear" data-clear hidden>Clear filters</button></p>
</div>` : "";

  const css = `
.cat h1{font:600 26px/1.2 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong);margin:0 0 6px}
.cat-lede{color:var(--anatomy-text);max-width:60ch;margin:0 0 22px;line-height:1.55}
.cat-links{margin:-8px 0 18px}
.cat-links a{font-weight:600;color:var(--anatomy-strong)}
.cat-filters{display:grid;gap:10px;padding:14px 16px;border:1px solid var(--anatomy-faint);border-radius:10px;background:var(--anatomy-card);margin-bottom:8px}
.cat-chips{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.cat-label{font:11px/1.4 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);min-width:88px}
.cat-chip{font:12px/1 ui-sans-serif,system-ui,sans-serif;padding:7px 10px;border-radius:999px;border:1px solid var(--anatomy-faint);background:none;color:var(--anatomy-text);cursor:pointer}
.cat-chip:hover{border-color:var(--anatomy-muted)}
.cat-chip[aria-pressed="true"]{background:var(--anatomy-strong);border-color:var(--anatomy-strong);color:var(--anatomy-page)}
.cat-chip:focus-visible,.cat-clear:focus-visible,.cat-search input:focus-visible{outline:2px solid var(--anatomy-strong);outline-offset:2px}
.cat-n{font:10px var(--anatomy-mono);opacity:.7;margin-left:2px}
.cat-search{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.cat-search input{flex:1;min-width:0;width:100%;max-width:360px;font:14px ui-sans-serif,system-ui,sans-serif;padding:8px 10px;border-radius:7px;border:1px solid var(--anatomy-faint);background:var(--anatomy-page);color:var(--anatomy-strong)}
.cat-result{margin:2px 0 0;font:12px var(--anatomy-mono);color:var(--anatomy-muted)}
.cat-clear{font:12px var(--anatomy-mono);background:none;border:0;padding:0 0 0 8px;color:var(--anatomy-strong);text-decoration:underline;cursor:pointer}
.cat-car h2{font:11px/1.4 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);margin:24px 0 10px}
.cat-jobs{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(min(300px,100%),1fr));gap:12px}
.cat-job{display:flex;flex-direction:column;gap:6px;height:100%;padding:16px;border:1px solid var(--anatomy-faint);border-radius:10px;background:var(--anatomy-card);color:var(--anatomy-text);text-decoration:none;transition:border-color .2s}
.cat-job:hover,.cat-job:focus-visible{border-color:var(--anatomy-strong)}
.cat-job strong{font:600 16px/1.3 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong)}
.cat-meta{font:11px/1.4 var(--anatomy-mono);color:var(--anatomy-muted)}
.cat-sum{line-height:1.5;font-size:13px}
.cat-problem{align-self:flex-start;font:11px/1.3 var(--anatomy-mono);padding:4px 7px;border-radius:5px;background:var(--anatomy-faint);color:var(--anatomy-strong)}
.cat-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;padding-top:6px}
.cat-tag{font:11px/1 var(--anatomy-mono);padding:5px 7px;border-radius:5px;border:1px solid var(--anatomy-faint);color:var(--anatomy-muted)}
.cat-tag[data-draft]{border-color:#a8691b;color:#a8691b}
.cat-tag[data-private]{border-color:var(--anatomy-strong);color:var(--anatomy-strong)}
.cat-empty{padding:28px 0;color:var(--anatomy-muted)}
.cat-foot{margin-top:36px;font-size:12px;color:var(--anatomy-muted);line-height:1.5;max-width:70ch}
@media (max-width:600px){.cat-label{min-width:100%}}`;

  const script = `
const root = document.querySelector(".cat");
const cards = [...root.querySelectorAll("[data-job]")];
const search = root.querySelector("[data-search]");
const result = root.querySelector("[data-result]");
const clear = root.querySelector("[data-clear]");
const empty = root.querySelector("[data-empty]");
// State: a set of picked values per chip group, plus the search text. Empty set = no filter.
const state = { type: new Set(), car: new Set(), system: new Set(), level: new Set(), time: new Set(), status: new Set(), q: "" };
function read() {
  const params = new URLSearchParams(location.hash.slice(1));
  for (const key of ["type", "car", "system", "level", "time", "status"]) state[key] = new Set((params.get(key) || "").split(",").filter(Boolean));
  state.q = params.get("q") || "";
}
function write() {
  const params = new URLSearchParams();
  for (const key of ["type", "car", "system", "level", "time", "status"]) if (state[key].size) params.set(key, [...state[key]].join(","));
  if (state.q) params.set("q", state.q);
  const hash = params.toString();
  history.replaceState(null, "", hash ? "#" + hash : location.pathname + location.search);
}
const timeOk = (minutes) => {
  const t = [...state.time][0];
  if (!t || t === "any") return true;
  return t === "long" ? minutes > 120 : minutes <= Number(t);
};
function apply() {
  const words = state.q.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const li of cards) {
    const d = li.dataset;
    const ok = (!state.type.size || state.type.has(d.type))
      && (!state.car.size || state.car.has(d.car))
      && (!state.system.size || state.system.has(d.system))
      && (!state.level.size || d.levels.split(" ").some((l) => state.level.has(l)))
      && timeOk(Number(d.minutes))
      && (!state.status.size || state.status.has(d.status))
      && words.every((w) => d.text.includes(w));
    li.hidden = !ok;
    if (ok) shown++;
  }
  for (const section of root.querySelectorAll("[data-car-section]")) section.hidden = !section.querySelector("[data-job]:not([hidden])");
  for (const group of root.querySelectorAll("[data-group]")) {
    for (const chip of group.querySelectorAll("[data-value]")) {
      const picked = group.dataset.group === "time" ? (state.time.size ? state.time.has(chip.dataset.value) : chip.dataset.value === "any") : state[group.dataset.group].has(chip.dataset.value);
      chip.setAttribute("aria-pressed", String(picked));
    }
  }
  const filtered = state.q || ["type", "car", "system", "level", "time", "status"].some((k) => state[k].size && !(k === "time" && state.time.has("any")));
  result.textContent = filtered ? (shown === 1 ? "1 job matches" : shown + " jobs match") : cards.length + " jobs";
  clear.hidden = !filtered;
  empty.hidden = shown > 0;
  if (search.value !== state.q) search.value = state.q;
}
root.addEventListener("click", (event) => {
  const chip = event.target.closest("[data-value]");
  if (chip) {
    const group = chip.closest("[data-group]");
    const set = state[group.dataset.group];
    const value = chip.dataset.value;
    if (group.dataset.multi === "1") set.has(value) ? set.delete(value) : set.add(value);
    else { set.clear(); if (value !== "any") set.add(value); }
    write(); apply();
  }
  if (event.target.closest("[data-clear]")) {
    for (const key of ["type", "car", "system", "level", "time", "status"]) state[key].clear();
    state.q = ""; write(); apply(); search.focus();
  }
});
search.addEventListener("input", () => { state.q = search.value.trim(); write(); apply(); });
window.addEventListener("hashchange", () => { read(); apply(); });
read(); apply();`;

  const body = `<div class="cat"><h1>${esc(title)}</h1><p class="cat-lede">Step-by-step trainers for car repairs. Practice a job on screen first, with the torques, sequences and checks quizzed, then take the Garage checklist to the car.</p>${symptoms ? `<p class="cat-links"><a href="symptoms.html">Something wrong with the car? Start from the symptom →</a></p>` : ""}${filters}${sections || "<p>No jobs in this build.</p>"}<p class="cat-empty" data-empty hidden>No jobs match these filters. <button type="button" class="cat-clear" data-clear>Clear filters</button></p><p class="cat-foot">Always check values against the source each job names. Jobs marked Draft have the procedure but no specs loaded yet.</p></div>`;
  return k.pageHtml({ title, theme, body: `<style>${css}</style>${body}`, script: jobs.length ? script : "", width: 1000 });
}

// The symptom guide: symptom → checks first → likely causes → links to the jobs in this build.
export function symptomsPage(symptoms, areas, jobs, { theme = "light" } = {}) {
  const byId = new Map(jobs.map((j) => [j.meta.id, j]));
  const link = (id) => (byId.has(id) ? `<a class="sy-job" href="${esc(byId.get(id).href)}">${esc(byId.get(id).meta.title)}</a>` : "");
  const card = (s) => {
    const causes = s.causes.map((c) => {
      const links = c.jobs.map(link).filter(Boolean);
      return `<li>${esc(c.text)}${links.length ? `<span class="sy-jobs">${links.join("")}</span>` : ""}</li>`;
    }).join("");
    const text = [s.symptom, ...s.first, ...s.causes.map((c) => c.text)].join(" ").toLowerCase();
    return `<article class="sy-card" id="${esc(s.id)}" data-sy data-text="${esc(text)}" ${s.stop ? "data-stop" : ""}>
  <h3>${esc(s.symptom)}</h3>
  ${s.stop ? `<p class="sy-stop">Stop driving until this is checked.</p>` : ""}
  <p class="sy-label">Check first</p><ul class="sy-first">${s.first.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
  <p class="sy-label">Likely causes</p><ul class="sy-causes">${causes}</ul>
</article>`;
  };
  const sections = areas.map((area) => {
    const list = symptoms.filter((s) => s.area === area);
    return list.length ? `<section class="sy-area" data-area><h2>${esc(area)}</h2>${list.map(card).join("")}</section>` : "";
  }).join("");
  const css = `
.sy h1{font:600 26px/1.2 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong);margin:0 0 6px}
.sy-back{font:12px var(--anatomy-mono);color:var(--anatomy-muted);text-decoration:none}
.sy-lede{color:var(--anatomy-text);max-width:62ch;line-height:1.55;margin:0 0 18px}
.sy-search{display:block;width:100%;max-width:420px;font:15px ui-sans-serif,system-ui,sans-serif;padding:10px 12px;border-radius:8px;border:1px solid var(--anatomy-faint);background:var(--anatomy-card);color:var(--anatomy-strong);margin-bottom:8px}
.sy-search:focus-visible{outline:2px solid var(--anatomy-strong);outline-offset:2px}
.sy-area h2{font:11px/1.4 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);margin:26px 0 10px}
.sy-card{border:1px solid var(--anatomy-faint);border-radius:10px;background:var(--anatomy-card);padding:14px 16px;margin-bottom:10px}
.sy-card[data-stop]{border-color:#b3412d}
.sy-card h3{font:600 16px/1.35 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong);margin:0 0 6px}
.sy-stop{margin:0 0 8px;font-weight:600;color:#b3412d}
.iso[data-theme="dark"] .sy-stop{color:#ff7b6b}
.sy-label{font:11px/1.4 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);margin:10px 0 4px}
.sy-card ul{margin:0;padding-left:18px;line-height:1.5}
.sy-causes li{margin-bottom:6px}
.sy-jobs{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.sy-job{font:12px/1 var(--anatomy-mono);padding:6px 8px;border-radius:6px;border:1px solid var(--anatomy-strong);color:var(--anatomy-strong);text-decoration:none}
.sy-job:hover,.sy-job:focus-visible{background:var(--anatomy-strong);color:var(--anatomy-page)}
.sy-none{color:var(--anatomy-muted)}
.sy-foot{margin-top:30px;font-size:12px;color:var(--anatomy-muted);line-height:1.5;max-width:70ch}`;
  const script = `
const q = document.querySelector("[data-q]");
const none = document.querySelector("[data-none]");
function apply() {
  const words = q.value.toLowerCase().split(/\\s+/).filter(Boolean);
  let shown = 0;
  for (const card of document.querySelectorAll("[data-sy]")) { const ok = words.every((w) => card.dataset.text.includes(w)); card.hidden = !ok; if (ok) shown++; }
  for (const area of document.querySelectorAll("[data-area]")) area.hidden = !area.querySelector("[data-sy]:not([hidden])");
  none.hidden = shown > 0;
}
q.addEventListener("input", apply);`;
  const body = `<div class="sy"><p><a class="sy-back" href="index.html">← All jobs</a></p><h1>Diagnose by symptom</h1><p class="sy-lede">Find what you notice, check the simple things first, then go to the job for each likely cause. These are common causes for the Audi A4 (B8) 2.0 TFSI, not a diagnosis: read the fault codes and check before you replace parts.</p><input type="search" class="sy-search" data-q placeholder="noise, brakes, warning light…" aria-label="Search symptoms">${sections}<p class="sy-none" data-none hidden>No symptoms match.</p><p class="sy-foot">Red cards mean stop driving until it's checked. Some causes link to no job yet.</p></div>`;
  return k.pageHtml({ title: "Diagnose by symptom", theme, body: `<style>${css}</style>${body}`, script, width: 820 });
}
