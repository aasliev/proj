# Car fixes

Step-by-step trainers for car repairs. Each repair is a **job**: practice it on screen
(questions, bolt-order puzzles, torque values), then use Garage mode or the printed
checklist at the car. The site is a catalog page plus one self-contained HTML page per job.

## Build and open

Needs Node.js 20 or later. From the repository root:

```
node site/build.mjs            # every job, private ones included (for your own use)
node site/build.mjs --public   # leaves out jobs marked private
node site/build.mjs --dark     # dark theme
```

Then open `site/dist/index.html` in a browser. `site/dist/symptoms.html` (linked from the
catalog) starts from what you notice, such as a knock over bumps or a warning light, and links to
the jobs for each likely cause. Its data is in `site/symptoms.mjs`; the build stops if it names a
job that doesn't exist, and a public build simply leaves out links to private jobs. The catalog filters by car, system, difficulty,
time and whether specs are loaded, plus a text search. The filters are kept in the address
(for example `index.html#system=Brakes&time=120`), so a filtered view can be bookmarked or shared. Each page in `site/dist/jobs/` also works on
its own. `site/dist/` is not committed: build it after cloning.

## Jobs

| Job | Status | Source |
|---|---|---|
| `ea888-caeb-assembly`: EA888 Gen 2 (CAEB) engine assembly, 70 steps | Ready, **private** | Audi repair manual (copyrighted) |
| `a4-b8-front-brakes`: Audi A4 B8 front pads and discs, 17 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-oil-change`: engine oil and filter change (2.0 TFSI), 10 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-brake-fluid`: brake fluid change, up to 10 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-fluid-check`: check and top up oil, coolant, brake and washer fluid, 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-wheel-swap`: wheel swap and tire check, up to 8 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-air-filter`: engine air filter, 4 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-cabin-filter`: cabin (pollen) filter, 4 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-wipers`: wiper blades, up to 6 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-battery`: battery test and replacement, 7 steps | **Draft** (voltage guide graded from general practice) | Licensed data, not loaded yet |
| `a4-b8-bulbs`: halogen headlamp bulb, 5 steps; xenon and LED stop at step 1 | **Draft** | Licensed data, not loaded yet |
| `a4-b8-annual-check`: whole-car inspection checklist, 8 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-rear-brakes`: rear pads and discs, electric or lever parking brake, up to 9 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-spark-plugs`: spark plugs and ignition coils (2.0 TFSI), 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-aux-belt`: auxiliary belt check and replacement, 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-jump-start`: jump start from a car or a jump pack, 6 steps | **Draft** | General practice and the owner's manual |
| `a4-b8-flat-tire`: spare wheel or repair kit at the roadside, up to 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-upper-control-arms`: front upper control arms, 6 steps | **Draft**, common problem | Licensed data, not loaded yet |
| `a4-b8-sway-bar-links`: anti-roll bar links, 3 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-shocks`: front shock absorbers, 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-carbon-cleaning`: intake valve carbon cleaning (2.0 TFSI), 5 steps | **Draft**, common problem | Licensed data, not loaded yet |
| `a4-b8-transmission-fluid`: transmission fluid service (CVT, automatic, dual-clutch or manual), up to 8 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-coolant-change`: coolant drain, refill and bleed, 5 steps | **Draft** | Licensed data, not loaded yet |
| `ea888-timing-chain-check`: valve timing check for chain stretch or a jumped tooth, 6 steps | Ready, **private**, common problem | Audi repair manual, Rep. Gr. 15 |
| `ea888-coolant-pump-leak`: pressure test, coolant pump and thermostat replacement, 9 steps | Ready, **private**, common problem | Audi repair manual, Rep. Gr. 19 |
| `ea888-pcv-oil-separator`: crankcase breather test and replacement, 3 steps | Ready, **private**, common problem | Audi repair manual, Rep. Gr. 17 |
| `ea888-compression-oil-pressure`: oil use, compression and oil pressure tests, 5 steps | Ready, **private**, common problem | Audi repair manual, Rep. Gr. 15 and 17 |

A **draft** has the full procedure, but no specs: torques show as "spec not loaded" and
measurements are recorded without a grade. Nothing in it is a guessed number.

A **private** job is built from material you may not republish. `--public` leaves it out;
never publish a build that includes it.

## Add a job

1. Make a folder `site/jobs/<id>/` with a `job.mjs` that exports:
   - `meta`: `id` (same as the folder name), `title`, `vehicle` `{make, model, generation, years}`,
     optional `engine`, `system` (Brakes, Engine, ...), `vehicleLine`, `status` (`draft` or
     `ready`), `private`, `minutes`, `difficulty`, `summary`,
     `source` `{name, prefix}` (prefix goes before each step's `ref`), optional `commonProblem` (shows a "Common problem" label and puts the job under the Type filter), `draftNote`, `statusLabel` (replaces "Draft · specs not loaded" on the card),
     `legal` and `storageKey`.
   - `build()`, returning `{steps, groups, variants, spec, diagrams, figures}`.
2. Draw its figures in a `figures.mjs` with the helpers in `lib/figure.mjs`. Every part sits in
   its own `data-part` group, with an offset where it waits until its step.
3. Run `node site/build.mjs`. The build stops with a message if a step names a figure,
   part, group, diagram or version that doesn't exist.

The fluid check (`jobs/a4-b8-fluid-check/job.mjs`) is the simplest example to copy. Jobs can
share drawings and bolt patterns: the wheel swap reuses the brake job's drawing, and
`lib/diagrams.mjs` holds the wheel star pattern.

A spec can also be marked not graded on purpose, with a reason shown in its place:
`{ pending: true, unit: "psi", why: "from the label inside the fuel door" }`.

### Step format

```js
{
  id: "caliper-off", group: "remove", title: "Take the caliper off",
  figure: "corner",            // which figure to show
  focus: ["caliper"],          // parts to highlight
  remove: ["caliper"],         // parts this step takes off (they slide away when done)
  install: [],                 // parts this step fits (they slide home when done)
  ref: "47 · p. 12",           // page in the source; "" shows the source's name
  when: { caliper: "floating" }, // only for this version
  say: "What to do and why.",
  tasks: [ ... ],
}
```

Task types:

| Type | Fields | Done when |
|---|---|---|
| `variant` | `keys`: version questions from `variants` | each is answered |
| `check` | `items`: strings (`"text · ref"` shows a reference) | all ticked |
| `choice` | `q`, `options: [{text, ok, why}]` | the right option is picked |
| `order` | `q`, `diagram`: a key of `diagrams` | bolts clicked in order |
| `torque` | `what`, `tool`, `note`, `stages: [{bolts, nm} \| {bolts, deg} \| {bolts, check: true} \| {bolts, pending: true}]` | each value entered or ticked |
| `measure` | `spec`: a key of `spec`, `label`, `items` | a reading for every item |
| `note` | `text` | always |

Any task can have its own `when`.

Specs: `{nominal, dev}`, `{min, max, limit}` (new range and wear limit above it), `{min, max, lowLimit}`
(wear limit below it, such as compression), `{min, max}`, or `{min}` alone, each with
`unit` and `ref`. A spec that depends on a version uses
`{unit, ref, byVariant: {key, values: {<option id>: {min, max, ...}}}}`. A `spec` id missing
from the table counts as not loaded: readings are recorded but not graded.

Diagrams: `{ref, shape, hint, bolts: [{n, x, y}]}` with positions in a 0–100 box and `shape`
one of `flange`, `cover`, `bracket`, `pan`, `wheel`; or a column of bearing caps,
`{ref, caps, side, label, hint, bolts: [{n, cap, side}]}`.

## Files

| Path | What it holds |
|---|---|
| `build.mjs` | Builds the catalog and every job page into `dist/` |
| `runtime/game.js`, `runtime/game.css` | The job engine shared by every job |
| `lib/page.mjs` | Page templates and the build checks |
| `lib/small-figures.mjs` | Drawings for the small jobs: filter box, wiper, battery, headlamp, ignition coils, belt drive, suspension, transmission |
| `symptoms.mjs` | The symptom guide's data |
| `lib/a4b8.mjs` | Shared settings for the A4 B8 draft jobs |
| `lib/diagrams.mjs` | Shared bolt patterns |
| `lib/figure.mjs` | Drawing helpers on top of the Anatomy kit (`.claude/skills/anatomy`) |
| `jobs/<id>/` | One job: `job.mjs`, and usually `figures.mjs` |

Progress is saved per job in the browser (`localStorage`), on that device only.
