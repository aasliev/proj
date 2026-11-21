// Builds the car fix site: a catalog page plus one self-contained page per job.
//   node site/build.mjs            all jobs, including private ones (for your own use)
//   node site/build.mjs --public   leaves out jobs marked private
//   node site/build.mjs --dark     dark theme
// Output: site/dist/index.html and site/dist/jobs/<id>.html. Open index.html in a browser.
import { mkdirSync, readdirSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { jobPage, catalogPage, symptomsPage } from "./lib/page.mjs";
import { SYMPTOMS, AREAS } from "./symptoms.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = process.argv.includes("--public");
const theme = process.argv.includes("--dark") ? "dark" : "light";
const OUT = join(HERE, "dist");

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "jobs"), { recursive: true });

const listed = [];
const allIds = new Set();
for (const dir of readdirSync(join(HERE, "jobs")).sort()) {
  const file = join(HERE, "jobs", dir, "job.mjs");
  if (!existsSync(file)) continue;
  const { meta, build } = await import(pathToFileURL(file).href);
  allIds.add(meta.id);
  if (meta.id !== dir) throw new Error(`jobs/${dir}: meta.id is "${meta.id}"; it must match the folder name`);
  if (PUBLIC && meta.private) {
    console.log("skip (private)", meta.id);
    continue;
  }
  const data = build();
  const page = jobPage(meta, data, { theme, back: "../index.html" });
  writeFileSync(join(OUT, "jobs", `${meta.id}.html`), page);
  listed.push({ meta, href: `jobs/${meta.id}.html`, steps: data.steps.length });
  console.log("wrote", `jobs/${meta.id}.html`, `${(page.length / 1024).toFixed(0)} KB`, `${data.steps.length} steps`, meta.status);
}
// Every job the symptom guide names must exist (it may be left out of a public build).
for (const s of SYMPTOMS) {
  if (!AREAS.includes(s.area)) throw new Error(`symptom ${s.id}: unknown area "${s.area}"`);
  for (const c of s.causes) for (const id of c.jobs) if (!allIds.has(id)) throw new Error(`symptom ${s.id}: no job "${id}"`);
}
writeFileSync(join(OUT, "symptoms.html"), symptomsPage(SYMPTOMS, AREAS, listed, { theme }));
writeFileSync(join(OUT, "index.html"), catalogPage(listed, { theme, symptoms: true }));
console.log("wrote index.html", `${listed.length} jobs`, PUBLIC ? "(public build)" : "(includes private jobs)");
