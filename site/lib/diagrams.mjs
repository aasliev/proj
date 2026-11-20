// Bolt patterns shared by several jobs. Positions are in a 0–100 box.

// Star order for a five-bolt wheel, starting at the top bolt: each bolt is followed by one
// across the hub. General workshop practice, not from a licensed source.
export function wheelStar() {
  const star = [0, 2, 4, 1, 3];
  return {
    ref: "general workshop practice",
    shape: "wheel",
    hint: "Go across the hub each time, not round it.",
    bolts: star.map((pos, i) => {
      const t = (-90 + pos * 72) * (Math.PI / 180);
      return { n: i + 1, x: 50 + 30 * Math.cos(t), y: 53 + 30 * Math.sin(t) };
    }),
  };
}
