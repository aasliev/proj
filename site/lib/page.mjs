// Builds one self-contained HTML page for a job: the job engine (runtime/game.js) with the
// job's data inlined. Also builds the catalogue page that links the jobs.
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

// The catalogue: jobs grouped by vehicle, then by system.
export function cataloguePage(jobs, { theme = "light", title = "Car fixes" } = {}) {
  const byCar = new Map();
  for (const { meta, href, steps } of jobs) {
    const v = meta.vehicle;
    const car = `${v.make} ${v.model}${v.generation ? ` (${v.generation})` : ""}`;
    if (!byCar.has(car)) byCar.set(car, []);
    byCar.get(car).push({ meta, href, steps });
  }
  const time = (m) => (m >= 120 ? `${Math.round(m / 60)} h` : `${m} min`);
  const cars = [...byCar].sort(([a], [b]) => a.localeCompare(b)).map(([car, list]) => `<section class="cat-car"><h2>${esc(car)}</h2><ul class="cat-jobs">${list.sort((a, b) => a.meta.system.localeCompare(b.meta.system) || a.meta.title.localeCompare(b.meta.title)).map(({ meta, href, steps }) => `<li><a class="cat-job" href="${esc(href)}"><span class="cat-meta">${esc(meta.system)}${meta.engine ? ` · ${esc(meta.engine)}` : ""}</span><strong>${esc(meta.title)}</strong><span class="cat-sum">${esc(meta.summary)}</span><span class="cat-tags">${meta.status === "draft" ? `<span class="cat-tag" data-draft>Draft · specs not loaded</span>` : `<span class="cat-tag">Specs loaded</span>`}<span class="cat-tag">${steps} steps</span><span class="cat-tag">about ${time(meta.minutes)}</span><span class="cat-tag">${esc(meta.difficulty)}</span>${meta.private ? `<span class="cat-tag" data-private>Private</span>` : ""}</span></a></li>`).join("")}</ul></section>`).join("");
  const css = `
.cat h1{font:600 26px/1.2 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong);margin:0 0 6px}
.cat-lede{color:var(--anatomy-text);max-width:60ch;margin:0 0 28px;line-height:1.55}
.cat-car h2{font:11px/1.4 var(--anatomy-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--anatomy-muted);margin:24px 0 10px}
.cat-jobs{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.cat-job{display:flex;flex-direction:column;gap:6px;height:100%;padding:16px;border:1px solid var(--anatomy-faint);border-radius:10px;background:var(--anatomy-card);color:var(--anatomy-text);text-decoration:none;transition:border-color .2s}
.cat-job:hover,.cat-job:focus-visible{border-color:var(--anatomy-strong)}
.cat-job strong{font:600 16px/1.3 ui-sans-serif,system-ui,sans-serif;color:var(--anatomy-strong)}
.cat-meta{font:11px/1.4 var(--anatomy-mono);color:var(--anatomy-muted)}
.cat-sum{line-height:1.5;font-size:13px}
.cat-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;padding-top:6px}
.cat-tag{font:11px/1 var(--anatomy-mono);padding:5px 7px;border-radius:5px;border:1px solid var(--anatomy-faint);color:var(--anatomy-muted)}
.cat-tag[data-draft]{border-color:#a8691b;color:#a8691b}
.cat-tag[data-private]{border-color:var(--anatomy-strong);color:var(--anatomy-strong)}
.cat-foot{margin-top:36px;font-size:12px;color:var(--anatomy-muted);line-height:1.5;max-width:70ch}`;
  const body = `<div class="cat"><h1>${esc(title)}</h1><p class="cat-lede">Step-by-step trainers for car repairs. Practise a job on screen first, with the torques, sequences and checks quizzed, then take the Garage checklist to the car.</p>${cars || "<p>No jobs in this build.</p>"}<p class="cat-foot">Always check values against the source each job names. Jobs marked Draft have the procedure but no specs loaded yet.</p></div>`;
  return k.pageHtml({ title, theme, body: `<style>${css}</style>${body}`, width: 1000 });
}
