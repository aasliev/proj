// EA888 Gen 2 (CAEB) rebuild trainer, stage 1: bottom-end assembly.
// Build: node docs/figures/ea888-caeb/rebuild/build.mjs [--dark]
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../../../../.claude/skills/anatomy/kit/iso-kit.mjs";
import { STEPS, GROUPS, VARIANTS, SPEC, DIAGRAMS } from "./steps.mjs";
import { buildFigures } from "./figures.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const THEME = process.argv.includes("--dark") ? "dark" : "light";
const figures = buildFigures();

// Every figure a step names must exist, and every part a step installs or lights must be drawn somewhere.
const drawn = new Set(Object.values(figures).flatMap((svg) => [...svg.matchAll(/data-part="([^"]+)"/g)].map((m) => m[1])));
for (const step of STEPS) {
  if (!figures[step.figure]) throw new Error(`${step.id}: no figure "${step.figure}"`);
  for (const name of [...(step.install || []), ...(step.focus || [])]) if (!drawn.has(name)) throw new Error(`${step.id}: no part "${name}" in any figure`);
}

const DATA = { steps: STEPS, groups: GROUPS, variants: VARIANTS, spec: SPEC, diagrams: DIAGRAMS, figures };
const CSS = readFileSync(join(HERE, "game.css"), "utf8");
const GAME = readFileSync(join(HERE, "game.js"), "utf8").replace("__DATA__", JSON.stringify(DATA));

const body = `<div class="rb iso-figure">
  <header class="rb-top">
    <div class="rb-title"><p class="rb-kicker">EA888 Gen 2 · CAEB · rebuild trainer</p><h1>Stage 1: bottom end</h1></div>
    <div class="rb-modes" role="group" aria-label="Mode"><button type="button" class="rb-mode" data-mode="game">Practice</button><button type="button" class="rb-mode" data-mode="garage">Garage</button></div>
    <div class="rb-progress"><div class="rb-bar" data-bar aria-hidden="true"></div><p class="rb-readout" data-readout aria-live="polite"></p></div>
  </header>
  <div class="rb-main">
    <div class="iso-plate rb-plate"><div class="rb-figure" data-figure></div></div>
    <article class="rb-card" data-card aria-live="polite"></article>
  </div>
  <nav class="rb-list" data-list aria-label="Steps"></nav>
  <div class="rb-actions"><button type="button" class="rb-btn" data-action="export">Export measurements</button><button type="button" class="rb-btn" data-action="print">Print checklist</button><button type="button" class="rb-btn" data-action="reset">Reset progress</button></div>
  <p class="rb-legal">Practice tool built from the Audi factory repair manual for the EA888 Gen II engine (Edition 06.2019). Every value names its repair group and page: check it against your copy before you turn a bolt. Progress and measurements are saved in this browser only; export them to keep a copy. Not affiliated with Audi.</p>
  <div class="rb-print" data-print></div>
</div>`;

const page = k.pageHtml({ title: "EA888 rebuild trainer", theme: THEME, body: `<style>${CSS}</style>${body}`, script: GAME, width: 1180 });
const out = THEME === "dark" ? "ea888-rebuild-dark.html" : "ea888-rebuild.html";
writeFileSync(join(HERE, out), page);
console.log("wrote", out, `${(page.length / 1024).toFixed(0)} KB`, `${STEPS.length} steps`);
