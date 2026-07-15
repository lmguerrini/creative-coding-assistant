"use client";

import {
  Activity,
  Boxes,
  Braces,
  Gauge,
  LayoutGrid,
  Paintbrush,
  Play,
  RefreshCw,
  Sparkles,
  type LucideIcon
} from "lucide-react";
import {
  demoModeRecommendedSequence,
  type DemoModeScenario
} from "@/lib/demo-mode";
import {
  DashboardCallout,
  DashboardCardGrid,
  DashboardChoiceCard,
  DashboardDefinitionGrid,
  DashboardDisclosure,
  DashboardPageHero,
  DashboardSection,
  DashboardSectionHeader
} from "./dashboard-page-primitives";

type DemoModePanelProps = {
  activeScenario: DemoModeScenario;
  hasImageAttachment: boolean;
  onLoadScenario: (scenario: DemoModeScenario) => void;
  onSelectScenario: (scenario: DemoModeScenario) => void;
  scenarios: readonly DemoModeScenario[];
  showDebugPanels: boolean;
};

export function DemoModePanel({
  activeScenario,
  hasImageAttachment,
  onLoadScenario,
  onSelectScenario,
  scenarios,
  showDebugPanels
}: DemoModePanelProps) {
  const imageAttachmentRequired = activeScenario.requiresImageAttachment === true;
  const canRunActiveScenario = !imageAttachmentRequired || hasImageAttachment;
  const selectedScenarioIcon = getDemoScenarioIcon(activeScenario);
  const productFacts = [
    { label: "Purpose", value: activeScenario.purpose },
    { label: "Runtime", value: activeScenario.runtime },
    { label: "Artifact", value: activeScenario.expectedArtifact },
    { label: "Preview", value: activeScenario.expectedPreview }
  ] as const;
  const demoWorkflow = [
    { label: "Input", value: activeScenario.inputRequirement },
    { label: "Interaction", value: activeScenario.expectedInteraction },
    { label: "Acceptance", value: activeScenario.expectedValidation },
    { label: "Fallback", value: activeScenario.fallback }
  ] as const;
  const technicalContract = [
    { label: "Concept", value: activeScenario.concept },
    { label: "Workflow", value: activeScenario.workflow },
    { label: "Generation", value: activeScenario.estimatedGenerationTime },
    { label: "Provider", value: activeScenario.providerRequirement },
    { label: "Retrieval", value: activeScenario.retrievalRequirement },
    { label: "Source boundary", value: activeScenario.sourceBoundary }
  ] as const;

  return (
    <section
      aria-label="Demo Mode"
      className="demoModePanel"
      data-debug={showDebugPanels ? "true" : "false"}
      id="demo-mode-panel"
    >
      <DashboardPageHero
        badgeLabel="Demo Mode coverage"
        badges={[`${scenarios.length} flows`, "4 creative runtimes", "Fallback-aware"]}
        className="demoModeHero"
        detail="Curated creative-coding journeys with explicit runtime, preview, interaction, and recovery boundaries."
        eyebrow="Demo Mode"
        headingLevel="h1"
        icon={Play}
        title="Creative scenarios"
      />

      {showDebugPanels ? (
        <DashboardSection
          className="demoFeaturedSequence"
          detail="Four complementary flows create a concise tour of supported runtime paths."
          eyebrow="Suggested sequence"
          icon={Sparkles}
          label="Featured demo paths"
          title="Recommended live sequence"
        >
          <DashboardCardGrid
            className="demoLiveSequence"
            label="Featured demo sequence"
            layout="quad"
            role="list"
          >
            {demoModeRecommendedSequence.map((item) => {
              const scenario = scenarios.find(
                (candidate) => candidate.id === item.scenarioId
              );
              if (!scenario) {
                return null;
              }

              return (
                <div key={`${item.role}-${item.scenarioId}`} role="listitem">
                  <DashboardChoiceCard
                    detail={item.rationale}
                    eyebrow={item.role}
                    icon={getDemoScenarioIcon(scenario)}
                    onClick={() => onSelectScenario(scenario)}
                    selected={scenario.id === activeScenario.id}
                    title={item.title}
                  />
                </div>
              );
            })}
          </DashboardCardGrid>
        </DashboardSection>
      ) : null}

      <DashboardSection
        className="demoScenarioWorkspace"
        detail="Choose one bounded flow, inspect its product contract, then load the prompt only when you are ready to run it."
        eyebrow="Scenario library"
        icon={LayoutGrid}
        label="Demo scenario workspace"
        title="Choose a showcase flow"
      >
        <div className="demoModeBody">
          <div
            aria-label="Demo Mode scenarios"
            className="demoScenarioList"
            role="list"
          >
            {scenarios.map((scenario) => (
              <div key={scenario.id} role="listitem">
                <DashboardChoiceCard
                  className="demoScenarioButton"
                  detail={showDebugPanels ? scenario.workflow : scenario.purpose}
                  eyebrow={getDemoScenarioPublicCategory(scenario)}
                  icon={getDemoScenarioIcon(scenario)}
                  onClick={() => onSelectScenario(scenario)}
                  selected={scenario.id === activeScenario.id}
                  title={scenario.title}
                />
              </div>
            ))}
          </div>

          <article
            aria-labelledby="demo-selected-scenario-label demo-selected-scenario-title"
            className="dashboardInnerCard demoScenarioDetail"
          >
            <span className="srOnly" id="demo-selected-scenario-label">
              Selected demo scenario
            </span>
            <DashboardSectionHeader
              action={(
                <button
                  aria-describedby={
                    !canRunActiveScenario ? "demo-image-required-notice" : undefined
                  }
                  className="dashboardPrimaryAction demoModeLoadButton"
                  disabled={!canRunActiveScenario}
                  onClick={() => onLoadScenario(activeScenario)}
                  type="button"
                >
                  <Play aria-hidden="true" size={15} />
                  <span>
                    {canRunActiveScenario ? "Load prompt & run" : "Attach image to run"}
                  </span>
                </button>
              )}
              className="demoScenarioDetailHeader"
              detail={activeScenario.description}
              eyebrow={getDemoScenarioPublicCategory(activeScenario)}
              icon={selectedScenarioIcon}
              title={activeScenario.title}
              titleId="demo-selected-scenario-title"
            />

            {imageAttachmentRequired && !hasImageAttachment ? (
              <div id="demo-image-required-notice" role="status">
                <DashboardCallout
                  detail="Add one image reference through the composer before running this demo."
                  icon={Paintbrush}
                  title="Image reference required"
                  tone="warning"
                />
              </div>
            ) : null}

            <DashboardDefinitionGrid
              className="demoScenarioQuickFacts"
              items={productFacts}
              label="Product essentials"
              layout="compact"
            />

            <DashboardDisclosure
              className="demoWorkflowDisclosure"
              defaultOpen={!showDebugPanels}
              summary="Demo workflow"
            >
              <DashboardDefinitionGrid
                items={demoWorkflow}
                label="Demo workflow details"
                layout="wide"
              />
            </DashboardDisclosure>

            <DashboardDisclosure
              className="demoPromptDisclosure"
              summary="Prompt preview"
            >
              <p className="demoPromptPreview">
                {showDebugPanels
                  ? activeScenario.prompt
                  : formatDemoPromptPreview(activeScenario.prompt)}
              </p>
            </DashboardDisclosure>

            {showDebugPanels ? (
              <DashboardDisclosure
                className="demoTechnicalDisclosure"
                summary="Technical contract"
              >
                <DashboardDefinitionGrid
                  items={technicalContract}
                  label="Technical demo contract"
                  layout="wide"
                />
              </DashboardDisclosure>
            ) : null}
          </article>
        </div>
      </DashboardSection>
    </section>
  );
}

function getDemoScenarioIcon(scenario: DemoModeScenario): LucideIcon {
  const signature = `${scenario.category} ${scenario.runtime}`.toLowerCase();

  if (signature.includes("tone") || signature.includes("audio")) {
    return Activity;
  }
  if (signature.includes("three") || signature.includes("3d")) {
    return LayoutGrid;
  }
  if (signature.includes("glsl") || signature.includes("shader")) {
    return Gauge;
  }
  if (signature.includes("export")) {
    return Boxes;
  }
  if (signature.includes("image") || signature.includes("multimodal")) {
    return Paintbrush;
  }
  if (signature.includes("failure") || signature.includes("recovery")) {
    return RefreshCw;
  }

  return Braces;
}

function getDemoScenarioPublicCategory(scenario: DemoModeScenario) {
  const searchable = [
    scenario.id,
    scenario.runtime,
    scenario.category,
    scenario.workflow,
    scenario.concept,
    scenario.title
  ]
    .join(" ")
    .toLowerCase();

  if (searchable.includes("tone") || searchable.includes("audio")) {
    return "Audio-visual";
  }
  if (searchable.includes("export") || searchable.includes("handoff")) {
    return "Export package";
  }
  if (searchable.includes("three")) {
    return "Three.js";
  }
  if (searchable.includes("p5")) {
    return "p5.js";
  }
  if (searchable.includes("glsl") || searchable.includes("shader")) {
    return "GLSL";
  }
  if (searchable.includes("retrieval") || searchable.includes("rag")) {
    return "Retrieval";
  }
  if (searchable.includes("agent")) {
    return "Agent workflow";
  }

  return "Creative workflow";
}

function formatDemoPromptPreview(prompt: string) {
  const normalizedPrompt = prompt.replace(/\s+/g, " ").trim();

  if (normalizedPrompt.length <= 170) {
    return normalizedPrompt;
  }

  return `${normalizedPrompt.slice(0, 167).trimEnd()}...`;
}
