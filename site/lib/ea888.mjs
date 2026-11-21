// Settings shared by jobs built from the Audi repair manual for the EA888 Gen II engine.
// These are PRIVATE: the manual is copyrighted, so public builds leave them out.
export const EA888_SOURCE = { name: "Audi repair manual, EA888 Gen II (Edition 06.2019)", prefix: "Repair manual, Rep. Gr. " };
export const EA888_LEGAL = "Built from the Audi factory repair manual for the EA888 Gen II engine (Edition 06.2019). Every value names its repair group and page: check it against your copy. The notes on common problems are what owners and workshops commonly report, not from the manual. Not affiliated with Audi.";

export function manualMeta(fields) {
  return {
    vehicle: { make: "Audi", model: "A4", generation: "B8", years: "2011" },
    engine: "EA888 Gen 2 · CAEB (2.0 TFSI)",
    vehicleLine: "EA888 Gen 2 · CAEB (2.0 TFSI), 2011 Audi A4",
    status: "ready",
    private: true,
    source: EA888_SOURCE,
    legal: EA888_LEGAL,
    ...fields,
  };
}
