// EA888 Gen 2 (CAEB): crankcase ventilation (PCV) oil separator on the cylinder head cover.
// Values from the Audi repair manual, Rep. Gr. 17, p. 164–166. PRIVATE.
import { manualMeta } from "../../lib/ea888.mjs";
import { ignition } from "../../lib/small-figures.mjs";
import { bay } from "../a4-b8-fluid-check/figures.mjs";

export const meta = manualMeta({
  id: "ea888-pcv-oil-separator",
  title: "Crankcase breather (PCV) oil separator",
  system: "Engine",
  commonProblem: "Failed crankcase breather (PCV) valve",
  minutes: 60,
  difficulty: "Beginner–intermediate",
  summary: "A failed breather diaphragm causes a whistle, rough idle, lean fault codes and oil use. Test for it, then replace the oil separator on the cylinder head cover.",
});

const GROUPS = [{ id: "diag", name: "Diagnose" }, { id: "swap", name: "Replace" }];

const STEPS = [
  {
    id: "symptoms",
    group: "diag",
    title: "Signs of a failed breather",
    figure: "bay",
    focus: ["oilCap"],
    ref: "",
    say: "The oil separator on top of the engine has a pressure control valve with a rubber diaphragm. Commonly reported when it tears: a whistle or squeal at idle, rough idle, lean mixture fault codes (for example P0171), oil use, and strong suction at the oil filler. A common quick test: with the engine idling, the oil filler cap is very hard to pull off, or a rubber glove over the filler is sucked in hard.",
    tasks: [
      { type: "check", items: ["Fault codes read and noted", "Whistle at idle listened for near the top of the engine", "Oil filler test done at idle (cap very hard to remove, or a glove sucked in hard = suspect the separator)"] },
      {
        type: "choice",
        q: "With the engine idling you can barely lift the oil filler cap, and it whistles. Most likely?",
        options: [
          { text: "Normal: engines always pull a vacuum on the filler", ok: false, why: "A slight vacuum is normal; one strong enough to hold the cap down is not." },
          { text: "The breather's diaphragm has failed", ok: true, why: "A torn diaphragm lets intake vacuum act on the crankcase. Replacing the separator is the usual fix." },
          { text: "The oil level is too high", ok: false, why: "Oil level doesn't cause strong vacuum at the filler." },
        ],
      },
    ],
  },
  {
    id: "remove",
    group: "swap",
    title: "Remove the oil separator from the head cover",
    figure: "ignition",
    focus: ["coils"],
    remove: ["coils"],
    ref: "17 · p. 166",
    say: "Take off the engine cover and the ignition coils for cylinders 3 and 4. Disconnect the crankcase ventilation hose, remove the separator's bolts and pull it off the ventilation hose. Cover the open valvetrain with a clean cloth.",
    tasks: [{ type: "check", items: ["Engine cover off (manual p. 11)", "Ignition coils 3 and 4 out · 28 · p. 254", "Ventilation hose disconnected", "Bolts out, separator pulled off the hose", "Valvetrain covered with a clean cloth"] }],
  },
  {
    id: "install",
    group: "swap",
    title: "Install the new separator",
    figure: "ignition",
    focus: ["coils"],
    install: ["coils"],
    ref: "17 · p. 164–166",
    say: "Fit the new separator with new seals. Its bolts cut their own thread in the head cover: only use the original-type bolts, and never cut the thread with a tap. Tighten in the manual's sequence, then refit coils 3 and 4.",
    tasks: [
      { type: "check", items: ["Cloth removed from the valvetrain", "New seals fitted (replace a seal after removing it) · 17 · p. 164", "Ventilation hose connected"] },
      { type: "torque", what: "Oil separator bolts on the cylinder head cover", note: "Thread-cutting bolts: original type only, no tap (17 · p. 165).", stages: [{ bolts: "Bolts 1–10, in the sequence in the manual's figure", nm: 11 }] },
      { type: "torque", what: "Ignition coil connector strip bolt", stages: [{ bolts: "Connector strip bolt", nm: 5 }] },
      { type: "check", items: ["Coils 3 and 4 pushed fully on, connectors clicked in", "Engine cover on", "Fault codes cleared; idle smooth; oil filler test repeated: only slight vacuum"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: {}, figures: { ignition: ignition(), bay: bay() } };
}
