// Audi A4 (B8) front upper control arms: a DRAFT job.
import { draftMeta, pending } from "../../lib/a4b8.mjs";
import { suspension } from "../../lib/small-figures.mjs";
import { corner } from "../a4-b8-front-brakes/figures.mjs";
import { wheelStar } from "../../lib/diagrams.mjs";

export const meta = draftMeta({
  id: "a4-b8-upper-control-arms",
  title: "Front upper control arms",
  system: "Suspension and steering",
  commonProblem: "Worn front upper control arms",
  minutes: 180,
  difficulty: "Intermediate",
  summary: "A knock or clunk from the front over bumps is commonly worn upper control arms on this car. Check them, replace both arms on a side, and tighten the bushings at ride height.",
});

const GROUPS = [{ id: "check", name: "Check" }, { id: "swap", name: "Replace" }, { id: "finish", name: "Finish" }];

const STEPS = [
  {
    id: "symptoms",
    group: "check",
    title: "Signs of worn upper arms",
    figure: "suspension",
    focus: ["upperArms"],
    ref: "",
    say: "Commonly reported on the B8: a knock or clunk from the front over bumps and potholes, vague steering, and uneven wear on the inner or outer edge of the front tires. The upper arms' ball joints and rubber bushings wear first.",
    tasks: [
      { type: "check", items: ["Noise noted: when, which side, over what kind of bump", "Front tires looked at for uneven edge wear"] },
      {
        type: "choice",
        q: "How do you check an upper arm ball joint for play?",
        options: [
          { text: "With the wheel off the ground, rock the wheel top-to-bottom by hand and watch the joint", ok: true, why: "Play at the top of the wheel that you can see at the upper arm's ball joint points to wear. A helper rocking while you watch makes it easy." },
          { text: "Listen for noise while driving only", ok: false, why: "Noise tells you something is worn; a hands-on check finds which part." },
          { text: "Push down on the fender", ok: false, why: "That tests the shock absorbers, not the ball joints." },
        ],
      },
      { type: "check", items: ["Wheel rocked top-to-bottom with the wheel off the ground: play at the upper ball joints noted", "Rubber boots on the ball joints checked for tears", "Bushings checked for cracks and grease leaks"] },
    ],
  },
  {
    id: "prep",
    group: "swap",
    title: "Parts, safety and wheel off",
    figure: "corner",
    focus: ["wheel", "bolts"],
    remove: ["bolts", "wheel"],
    ref: "",
    say: "Upper arms are usually replaced as a pair on a side (front and rear arm), often both sides together. Many of the bolts are single-use: get new ones. Raise the car on stands and take the wheel off.",
    tasks: [{ type: "check", items: ["Two new upper arms for the side (or a full kit), for your car's suspension version", "New bolts and nuts your source lists as single-use (spec not loaded)", "Car on axle stands, wheel off", "Floor jack ready under the lower arm to support the wheel bearing housing"] }],
  },
  {
    id: "remove",
    group: "swap",
    title: "Remove the upper arms",
    figure: "suspension",
    focus: ["upperArms", "knuckle"],
    remove: ["upperArms"],
    ref: "",
    say: "Take the load off the arms by raising the jack gently under the lower arm. Undo the clamp bolt that holds the ball joints in the wheel bearing housing and free the joints. Then remove the bolts that hold the arms to the body as your source describes; on this car their access is usually from the top of the wheel house or under the hood.",
    tasks: [
      { type: "check", items: ["Wheel bearing housing supported on the jack", "Brake hose and sensor wires not stretched as the housing moves", "Clamp bolt out, ball joints freed without damaging the housing", "Body-side bolts out; arms out"] },
      {
        type: "choice",
        q: "The ball joint won't come out of the housing. What do you do?",
        options: [
          { text: "Hit the wheel bearing housing with a big hammer", ok: false, why: "That can crack the aluminum housing. Spread the clamp slot with the right tool, as your source describes." },
          { text: "Open the clamp slot slightly with a spreader tool and lever the joint out", ok: true, why: "The joint is held by the clamp; spreading it a little releases it without damage." },
          { text: "Cut the arm off", ok: false, why: "Not needed, and risks damaging the housing." },
        ],
      },
    ],
  },
  {
    id: "install",
    group: "swap",
    title: "Fit the new arms",
    figure: "suspension",
    focus: ["upperArms"],
    install: ["upperArms"],
    ref: "",
    say: "Fit the new arms to the body with their bolts loose, then the ball joints into the wheel bearing housing and the new clamp bolt. Tighten the clamp bolt now. Leave the body-side bolts loose: rubber bushings must be tightened with the suspension at its normal ride height, or they will be twisted and wear quickly.",
    tasks: [
      { type: "check", items: ["Arms on the body, bolts fitted but loose", "Ball joints fully seated in the housing"] },
      { type: "torque", what: "Ball joint clamp bolt (new)", stages: [pending("Clamp bolt")] },
    ],
  },
  {
    id: "ride-height",
    group: "finish",
    title: "Tighten the bushings at ride height",
    figure: "suspension",
    focus: ["upperArms", "lowerArms"],
    ref: "",
    say: "Raise the suspension with the jack until the car just starts to lift off the stand, or set the ride height your source gives by measurement. Then tighten the body-side bolts.",
    tasks: [
      { type: "check", items: ["Suspension at ride height (the car's weight on it, or set to the height your source specifies)"] },
      { type: "torque", what: "Upper arm body-side bolts", note: "Some of these take a torque and then a further angle: follow your source.", stages: [pending("Body-side bolts"), { bolts: "Further angle, only if your source gives one", check: true }] },
      {
        type: "choice",
        q: "Why not tighten the bushing bolts with the wheel hanging?",
        options: [
          { text: "It makes no difference", ok: false, why: "It does: the bushing is clamped in its hanging position." },
          { text: "The bushing would be twisted at ride height all the time, and wear out quickly", ok: true, why: "Tightening at ride height means the rubber is relaxed where the car spends its time." },
          { text: "The bolts are easier to reach", ok: false, why: "Access isn't the reason." },
        ],
      },
    ],
  },
  {
    id: "finish",
    group: "finish",
    title: "Wheel on and alignment",
    figure: "corner",
    focus: ["wheel", "bolts"],
    install: ["wheel", "bolts"],
    ref: "",
    say: "Refit the wheel, tighten in a star pattern and test drive. New arms change the wheel alignment: have it checked and set afterwards.",
    tasks: [
      { type: "order", diagram: "wheel", q: "Tighten the wheel bolts in a star pattern, starting with the top bolt." },
      { type: "torque", what: "Wheel bolts", tool: "torque wrench", stages: [pending("All five, in the star pattern")] },
      { type: "check", items: ["Test drive: no knocks, steering straight", "Wheel alignment checked and set at a shop"] },
    ],
  },
];

export function build() {
  return { steps: STEPS, groups: GROUPS, variants: {}, spec: {}, diagrams: { wheel: wheelStar() }, figures: { suspension: suspension(), corner: corner("A front wheel corner with the wheel and its bolts.") } };
}
