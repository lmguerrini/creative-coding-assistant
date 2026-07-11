import {
  demoModeScenarios,
  type DemoModeScenario
} from "./demo-mode";

export type DemoValidationLayer =
  | "contract"
  | "automated"
  | "provider"
  | "visible_output"
  | "human_acceptance";

export type DemoFixtureKind =
  | "browser_runtime"
  | "answer_only"
  | "export_only"
  | "image_reference"
  | "controlled_failure";

export type GoldenDemoFixture = {
  scenarioId: DemoModeScenario["id"];
  fixtureKind: DemoFixtureKind;
  requiredLayers: DemoValidationLayer[];
  prompt: string;
  expectedArtifact: string;
  fallback: string;
};

export type DemoDurationBudget = {
  scenarioId: DemoModeScenario["id"];
  generationSeconds: number;
  inspectionSeconds: number;
};

export type DemoClarificationFixture = {
  scenarioId: DemoModeScenario["id"];
  question: string;
  options: string[];
  expectedSelection: string;
};

export type DemoReliabilitySample = {
  scenarioId: DemoModeScenario["id"];
  layer: Exclude<DemoValidationLayer, "human_acceptance">;
  passed: boolean;
};

export type DemoQualitySample = {
  scenarioId: DemoModeScenario["id"];
  craft: number;
  clarity: number;
  safety: number;
  truthfulness: number;
};

export const goldenDemoFixtures: GoldenDemoFixture[] = demoModeScenarios.map(
  (scenario) => ({
    scenarioId: scenario.id,
    fixtureKind: fixtureKindForScenario(scenario),
    requiredLayers: requiredLayersForScenario(scenario),
    prompt: scenario.prompt,
    expectedArtifact: scenario.expectedArtifact,
    fallback: scenario.fallback
  })
);

export const demoDurationBudgets: DemoDurationBudget[] = [
  { scenarioId: "cymatic-chladni-audiovisual", generationSeconds: 90, inspectionSeconds: 30 },
  { scenarioId: "physarum-p5-hero", generationSeconds: 90, inspectionSeconds: 30 },
  { scenarioId: "kinetic-three-hero", generationSeconds: 90, inspectionSeconds: 45 },
  { scenarioId: "chladni-glsl-hero", generationSeconds: 90, inspectionSeconds: 30 },
  { scenarioId: "retrieval-grounded-design-brief", generationSeconds: 60, inspectionSeconds: 30 },
  { scenarioId: "multi-agent-production-plan", generationSeconds: 90, inspectionSeconds: 30 },
  { scenarioId: "single-agent-line-study", generationSeconds: 90, inspectionSeconds: 30 },
  { scenarioId: "export-handoff-package", generationSeconds: 60, inspectionSeconds: 45 },
  { scenarioId: "multimodal-reference-study", generationSeconds: 90, inspectionSeconds: 45 },
  { scenarioId: "failure-recovery-rehearsal", generationSeconds: 30, inspectionSeconds: 30 }
];

export const demoClarificationFixtures: DemoClarificationFixture[] = [
  {
    scenarioId: "multimodal-reference-study",
    question: "What should guide the palette study when no reference image is attached?",
    options: ["Use a textual palette brief", "Fetch an image", "Skip the fallback"],
    expectedSelection: "Use a textual palette brief"
  },
  {
    scenarioId: "failure-recovery-rehearsal",
    question: "What should happen when the provider is unavailable?",
    options: ["Keep the local draft and retry later", "Claim the preview ran", "Clear the session"],
    expectedSelection: "Keep the local draft and retry later"
  }
];

export function auditDemoScenarioMetadata(
  scenarios: readonly DemoModeScenario[] = demoModeScenarios
) {
  const issues: string[] = [];
  const forbiddenClaims = /\b(?:sacred|spiritual|therapeutic|medical|autonomous delivery)\b/i;

  for (const scenario of scenarios) {
    const required: Array<[string, string]> = [
      ["title", scenario.title],
      ["concept", scenario.concept],
      ["purpose", scenario.purpose],
      ["runtime", scenario.runtime],
      ["workflow", scenario.workflow],
      ["input", scenario.inputRequirement],
      ["prompt", scenario.prompt],
      ["artifact", scenario.expectedArtifact],
      ["preview", scenario.expectedPreview],
      ["interaction", scenario.expectedInteraction],
      ["validation", scenario.expectedValidation],
      ["fallback", scenario.fallback],
      ["source boundary", scenario.sourceBoundary]
    ];

    for (const [field, value] of required) {
      if (!value.trim()) {
        issues.push(`${scenario.id}: missing ${field}.`);
      }
    }

    if (forbiddenClaims.test(`${scenario.prompt}\n${scenario.sourceBoundary}`)) {
      issues.push(`${scenario.id}: contains a prohibited product claim.`);
    }
  }

  const scenarioIds = scenarios.map((scenario) => scenario.id);
  if (new Set(scenarioIds).size !== scenarioIds.length) {
    issues.push("Demo scenario ids must be unique.");
  }

  return issues;
}

export function validateDemoPromptContracts(
  scenarios: readonly DemoModeScenario[] = demoModeScenarios
) {
  const issues: string[] = [];

  for (const scenario of scenarios) {
    const fixtureKind = fixtureKindForScenario(scenario);
    if (fixtureKind === "browser_runtime" && !/return only/i.test(scenario.prompt)) {
      issues.push(`${scenario.id}: browser-runtime prompt must request an artifact only.`);
    }
    if (fixtureKind === "controlled_failure" && !/do not claim/i.test(scenario.prompt)) {
      issues.push(`${scenario.id}: failure rehearsal must prohibit a false preview claim.`);
    }
    if (fixtureKind === "image_reference" && !/attached image/i.test(scenario.prompt)) {
      issues.push(`${scenario.id}: image-reference prompt must state its input boundary.`);
    }
  }

  return issues;
}

export function summarizeDemoReliability(samples: readonly DemoReliabilitySample[]) {
  const passed = samples.filter((sample) => sample.passed).length;
  const total = samples.length;
  return {
    passed,
    total,
    passRate: total === 0 ? 0 : passed / total,
    failedScenarioIds: [...new Set(samples.filter((sample) => !sample.passed).map((sample) => sample.scenarioId))]
  };
}

export function summarizeDemoOutputQuality(samples: readonly DemoQualitySample[]) {
  const total = samples.length;
  const average = (field: keyof Omit<DemoQualitySample, "scenarioId">) =>
    total === 0
      ? 0
      : samples.reduce((sum, sample) => sum + sample[field], 0) / total;
  const scores = {
    craft: average("craft"),
    clarity: average("clarity"),
    safety: average("safety"),
    truthfulness: average("truthfulness")
  };

  return {
    ...scores,
    overall: (scores.craft + scores.clarity + scores.safety + scores.truthfulness) / 4
  };
}

export function totalDemoDurationSeconds(
  budgets: readonly DemoDurationBudget[] = demoDurationBudgets
) {
  return budgets.reduce(
    (total, budget) => total + budget.generationSeconds + budget.inspectionSeconds,
    0
  );
}

function fixtureKindForScenario(scenario: DemoModeScenario): DemoFixtureKind {
  if (scenario.id === "failure-recovery-rehearsal") {
    return "controlled_failure";
  }
  if (scenario.id === "multimodal-reference-study") {
    return "image_reference";
  }
  if (scenario.id === "export-handoff-package") {
    return "export_only";
  }
  return "browser_runtime";
}

function requiredLayersForScenario(scenario: DemoModeScenario): DemoValidationLayer[] {
  const shared: DemoValidationLayer[] = ["contract", "automated", "human_acceptance"];
  const kind = fixtureKindForScenario(scenario);

  if (kind === "controlled_failure") {
    return [...shared, "visible_output"];
  }
  if (kind === "answer_only" || kind === "export_only") {
    return [...shared, "provider", "visible_output"];
  }
  return [...shared, "provider", "visible_output"];
}
