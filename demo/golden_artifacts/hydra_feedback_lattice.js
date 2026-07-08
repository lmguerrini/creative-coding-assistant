(function attachHydraFeedbackLattice(root) {
  function createHydraFeedbackLattice(hydra) {
    if (!hydra || typeof hydra.osc !== "function" || typeof hydra.noise !== "function") {
      throw new Error("Hydra synth scope is unavailable.");
    }

    hydra.speed = 0.72;
    hydra
      .osc(9, 0.08, 1.35)
      .kaleid(6)
      .color(1.2, 0.58, 1.65)
      .rotate(0.12, 0.04)
      .modulate(hydra.noise(3.1, 0.16), 0.075)
      .contrast(1.18)
      .out(hydra.o0);
    hydra.render(hydra.o0);
  }

  root.createHydraFeedbackLattice = createHydraFeedbackLattice;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createHydraFeedbackLattice };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
