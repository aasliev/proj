// Settings shared by the Audi A4 (B8) draft jobs.
export const A4B8 = { make: "Audi", model: "A4", generation: "B8", years: "2008–2016" };
export const LICENSED_PENDING = { name: "Licensed repair data (not loaded yet)", prefix: "Source: " };

export function draftMeta(fields) {
  return {
    vehicle: A4B8,
    vehicleLine: "Audi A4 (B8, 2008–2016)",
    status: "draft",
    private: false,
    source: LICENSED_PENDING,
    draftNote: "The procedure is general workshop practice. No licensed data is loaded yet, so part numbers, intervals and torques show as “spec not loaded”: take them from your licensed data or the owner's manual.",
    ...fields,
  };
}
export const pending = (bolts) => ({ bolts, pending: true });
