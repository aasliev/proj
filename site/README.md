# Car fixes

Step-by-step trainers for car repairs. Each repair is a **job**: practise it on screen
(questions, bolt-order puzzles, torque values), then use Garage mode or the printed
checklist at the car. The site is a catalogue page plus one self-contained HTML page per job.

## Build and open

Needs Node.js 20 or later. From the repository root:

```
node site/build.mjs            # every job, private ones included (for your own use)
node site/build.mjs --public   # leaves out jobs marked private
node site/build.mjs --dark     # dark theme
```

Then open `site/dist/index.html` in a browser. Each page in `site/dist/jobs/` also works on
its own. `site/dist/` is not committed: build it after cloning.

## Jobs

| Job | Status | Source |
|---|---|---|
| `ea888-caeb-assembly`: EA888 Gen 2 (CAEB) engine assembly, 70 steps | Ready, **private** | Audi repair manual (copyrighted) |
| `a4-b8-front-brakes`: Audi A4 B8 front pads and discs, 17 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-oil-change`: engine oil and filter change (2.0 TFSI), 10 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-brake-fluid`: brake fluid change, up to 10 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-fluid-check`: check and top up oil, coolant, brake and washer fluid, 5 steps | **Draft** | Licensed data, not loaded yet |
| `a4-b8-wheel-swap`: wheel swap and tyre check, up to 8 steps | **Draft** | Licensed data, not loaded yet |

A **draft** has the full procedure, but no specs: torques show as "spec not loaded" and
measurements are recorded without a grade. Nothing in it is a guessed number.

A **private** job is built from material you may not republish. `--public` leaves it out;
never publish a build that includes it.

## Add a job

1. Make a folder `site/jobs/<id>/` with a `job.mjs` that exports:
   - `meta`: `id` (same as the folder name), `title`, `vehicle` `{make, model, generation, years}`,
     optional `engine`, `system` (Brakes, Engine, ...), `vehicleLine`, `status` (`draft` or
     `ready`), `private`, `minutes`, `difficulty`, `summary`,
     `source` `{name, prefix}` (prefix goes before each step's `ref`), optional `draftNote`,
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
`{ pending: true, unit: "bar", why: "from the label inside the fuel filler flap" }`.

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

Specs: `{nominal, dev}`, `{min, max, limit}` (new range and wear limit), or `{min, max}`, each with
`unit` and `ref`. A spec that depends on a version uses
`{unit, ref, byVariant: {key, values: {<option id>: {min, max, ...}}}}`. A `spec` id missing
from the table counts as not loaded: readings are recorded but not graded.

Diagrams: `{ref, shape, hint, bolts: [{n, x, y}]}` with positions in a 0–100 box and `shape`
one of `flange`, `cover`, `bracket`, `pan`, `wheel`; or a column of bearing caps,
`{ref, caps, side, label, hint, bolts: [{n, cap, side}]}`.

## Files

| Path | What it holds |
|---|---|
| `build.mjs` | Builds the catalogue and every job page into `dist/` |
| `runtime/game.js`, `runtime/game.css` | The job engine shared by every job |
| `lib/page.mjs` | Page templates and the build checks |
| `lib/figure.mjs` | Drawing helpers on top of the Anatomy kit (`.claude/skills/anatomy`) |
| `jobs/<id>/` | One job: `job.mjs`, and usually `figures.mjs` |

Progress is saved per job in the browser (`localStorage`), on that device only.
