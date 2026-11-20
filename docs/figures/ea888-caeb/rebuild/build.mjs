// EA888 Gen 2 (CAEB) rebuild trainer as one stand-alone page, built with the site's shared job
// engine (site/runtime) from the job in site/jobs/ea888-caeb-assembly.
// Build: node docs/figures/ea888-caeb/rebuild/build.mjs [--dark]
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { meta, build } from "../../../../site/jobs/ea888-caeb-assembly/job.mjs";
import { jobPage } from "../../../../site/lib/page.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const THEME = process.argv.includes("--dark") ? "dark" : "light";
const data = build();
const page = jobPage(meta, data, { theme: THEME });
const out = THEME === "dark" ? "ea888-rebuild-dark.html" : "ea888-rebuild.html";
writeFileSync(join(HERE, out), page);
console.log("wrote", out, `${(page.length / 1024).toFixed(0)} KB`, `${data.steps.length} steps`);
