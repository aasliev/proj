// Builds the car fix site: a catalogue page plus one self-contained page per job.
//   node site/build.mjs            all jobs, including private ones (for your own use)
//   node site/build.mjs --public   leaves out jobs marked private
//   node site/build.mjs --dark     dark theme
// Output: site/dist/index.html and site/dist/jobs/<id>.html. Open index.html in a browser.
import { mkdirSync, readdirSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { jobPage, cataloguePage } from "./lib/page.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = process.argv.includes("--public");
const theme = process.argv.includes("--dark") ? "dark" : "light";
const OUT = join(HERE, "dist");

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, "jobs"), { recursive: true });

const listed = [];
for (const dir of readdirSync(join(HERE, "jobs")).sort()) {
  const file = join(HERE, "jobs", dir, "job.mjs");
  if (!existsSync(file)) continue;
  const { meta, build } = await import(pathToFileURL(file).href);
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
writeFileSync(join(OUT, "index.html"), cataloguePage(listed, { theme }));
console.log("wrote index.html", `${listed.length} jobs`, PUBLIC ? "(public build)" : "(includes private jobs)");
