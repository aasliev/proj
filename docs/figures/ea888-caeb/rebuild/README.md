# EA888 Gen 2 (CAEB) rebuild trainer

A step-by-step practice game and workshop checklist for assembling the 2.0 TFSI
engine (code CAEB, 2011 Audi A4): 70 steps from the bare block through the
cylinder head, timing chains, covers, oil pans and coolant pump. Every torque,
clearance and sequence names its page in the Audi repair manual.

## Open it

No install or server needed.

1. Clone the repository.
2. Open `ea888-rebuild.html` in this folder in Chrome, Edge, Firefox or Safari
   (double-click it). `ea888-rebuild-dark.html` is the same game with a dark theme.

Opening the file on github.com only shows its source code; clone or download it first.

## Use it

- **Practice** mode quizzes you: answer the questions, click bolts in tightening
  order, type in torque and angle values. The next step unlocks when the current
  one is done.
- **Garage** mode shows every spec plainly, as a checklist to tick off at the bench.
- Step 1 asks which versions of some parts your engine has (rod bolt size, oil ring
  type, flange and cover bolt counts, head bolt type and so on), and whether the
  balance shafts came out of the block. Check your parts: the torques depend on
  them. The balance shafts and their bolts are only replaced if they were removed
  (Rep. Gr. 13, p. 43); if they stayed in, their steps are skipped.
- Measurement steps grade each reading against the manual's new-part range and
  wear limit. Enter your real readings, or use "Fill practice readings".
- Progress and measurements are saved in your browser on this computer only.
  Use **Export measurements** to keep a copy, and **Print checklist** for paper.

## Rebuild after editing

Needs Node.js 20 or later. From the repository root:

```
node docs/figures/ea888-caeb/rebuild/build.mjs          # ea888-rebuild.html
node docs/figures/ea888-caeb/rebuild/build.mjs --dark   # ea888-rebuild-dark.html
```

| File | What it holds |
|---|---|
| `steps.mjs` | Stage 1, bottom end; splices in the other stages in assembly order |
| `steps-stage2.mjs` | Stage 2, timing drive, covers, oil pans, coolant pump |
| `steps-stage3.mjs` | Stage 3, cylinder head, valvetrain, camshafts |
| `figures.mjs` | The drawings, built with the Anatomy skill in `.claude/skills/anatomy` |
| `build.mjs` | Puts it all into one HTML file |

The game itself is the shared job engine in `site/runtime/`, and the job's settings
(title, source, version-dependent ring specs) are in `site/jobs/ea888-caeb-assembly/job.mjs`.
The same trainer is also a page of the car fixes site: see `site/README.md`.

## Important

- **Check every value against your own manual** before you turn a bolt. The steps
  are written in our own words and the drawings are simplified.
- **Keep this private.** It is built from Audi's copyrighted repair manual (EA888
  Gen II, Edition 06.2019). Share it only with people who own the manual. Do not
  publish it, put it in a public repository or turn on GitHub Pages for it.
- Not covered yet: disassembly, and the bolt-on parts (intake manifold, fuel
  system, turbocharger, oil filter bracket, belt drive, coils and wiring).
