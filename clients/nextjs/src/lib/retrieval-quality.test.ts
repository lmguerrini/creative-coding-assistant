import { describe, expect, it } from "vitest";
import {
  getInitialWorkspaceSnapshot,
  getLocalWorkspaceSnapshot
} from "./assistant-client";
import { buildRetrievalQualityModel } from "./retrieval-quality";
import { buildRetrievalRuntimeModel } from "./retrieval-runtime";

describe("retrieval quality", () => {
  it("explains precision, diversity, coverage, sufficiency, and domain balance", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const quality = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    expect(quality).toMatchObject({
      overallLevel: "medium",
      overallLabel: "Medium retrieval quality",
      overallDetail:
        "Quality is medium because precision remains the limiting signal.",
      hasEvidence: true,
      domainBalance: {
        status: "balanced",
        label: "Balanced across 2 domains",
        detail: "WebGPU / WGSL contributes 67% of context."
      }
    });
    expect(quality.metrics).toEqual([
      expect.objectContaining({
        key: "precision",
        level: "medium",
        valueLabel: "83% average",
        detail:
          "3 of 3 selected chunks scored; 1 clears the 85% high-confidence threshold. This is a relevance-score proxy, not offline evaluation precision."
      }),
      expect.objectContaining({
        key: "diversity",
        level: "high",
        valueLabel: "2 sources · 2 domains"
      }),
      expect.objectContaining({
        key: "coverage",
        level: "high",
        valueLabel: "2/2 requested domains"
      }),
      expect.objectContaining({
        key: "context_sufficiency",
        level: "high",
        valueLabel: "3 chunks selected"
      })
    ]);
    expect(quality.domainBalance.domains).toEqual([
      {
        domain: "webgpu_wgsl",
        label: "WebGPU / WGSL",
        chunkCount: 2,
        sharePercent: 67,
        shareLabel: "67% of context",
        requested: true
      },
      {
        domain: "glsl",
        label: "GLSL",
        chunkCount: 1,
        sharePercent: 33,
        shareLabel: "33% of context",
        requested: true
      }
    ]);
    expect(quality.weaknesses).toEqual([
      "Average selected-chunk relevance is moderate rather than high.",
      "1 selected source may be stale."
    ]);
  });

  it("identifies low-quality concentration and missing domain coverage", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const retrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source, sourceIndex) => ({
        ...source,
        selectedForContext: sourceIndex === 0,
        chunks: source.chunks.map((chunk, chunkIndex) => ({
          ...chunk,
          score: sourceIndex === 0 ? 0.64 - chunkIndex * 0.04 : chunk.score,
          usedInContext: sourceIndex === 0
        }))
      }))
    };
    const quality = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(retrieval, [])
    );

    expect(quality).toMatchObject({
      overallLevel: "low",
      overallLabel: "Low retrieval quality",
      overallDetail:
        "Quality is low because precision and context sufficiency are below the reliable range.",
      domainBalance: {
        status: "concentrated",
        label: "Concentrated in WebGPU / WGSL",
        detail: "1 requested domain did not contribute selected context."
      }
    });
    expect(quality.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "precision",
          level: "low",
          valueLabel: "62% average"
        }),
        expect.objectContaining({
          key: "diversity",
          level: "low",
          valueLabel: "1 source · 1 domain"
        }),
        expect.objectContaining({
          key: "coverage",
          level: "medium",
          valueLabel: "1/2 requested domains",
          detail: "Selected context is missing GLSL."
        }),
        expect.objectContaining({
          key: "context_sufficiency",
          level: "low"
        })
      ])
    );
    expect(quality.weaknesses).toEqual([
      "Average selected-chunk relevance is below 70%.",
      "Selected context is missing GLSL.",
      "Generation context depends on a single source."
    ]);
  });

  it("uses safe quality fallbacks for legacy retrieval sessions", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const legacyRetrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source) => ({
        ...source,
        selectedForContext: undefined,
        chunks: source.chunks.map((chunk) => ({
          ...chunk,
          score: null,
          usedInContext: undefined
        }))
      }))
    };
    const quality = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(legacyRetrieval, [])
    );

    expect(quality).toMatchObject({
      overallLevel: "medium",
      overallDetail:
        "Quality is medium because precision remains the limiting signal.",
      hasEvidence: true
    });
    expect(quality.metrics[0]).toEqual(
      expect.objectContaining({
        key: "precision",
        level: "unknown",
        valueLabel: "Not scored",
        detail:
          "Selected chunks have no relevance scores, so score-based precision cannot be verified."
      })
    );
    expect(quality.weaknesses).toContain(
      "Relevance scores were not recorded, so precision cannot be verified."
    );
    expect(JSON.stringify(quality)).not.toMatch(/NaN|Infinity/);
  });

  it("does not overstate precision when only part of the context is scored", () => {
    const snapshot = getLocalWorkspaceSnapshot();
    const partialScoreRetrieval = {
      ...snapshot.retrieval,
      sources: snapshot.retrieval.sources.map((source, sourceIndex) => ({
        ...source,
        chunks: source.chunks.map((chunk, chunkIndex) => ({
          ...chunk,
          score: sourceIndex === 1 && chunkIndex === 0 ? null : chunk.score
        }))
      }))
    };
    const quality = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(partialScoreRetrieval, [])
    );

    expect(quality.metrics[0]).toEqual(
      expect.objectContaining({
        key: "precision",
        level: "medium",
        valueLabel: "88% average",
        detail:
          "2 of 3 selected chunks scored; 1 clears the 85% high-confidence threshold. This is a relevance-score proxy, not offline evaluation precision."
      })
    );
    expect(quality.weaknesses).toContain(
      "Only 2 of 3 selected chunks include relevance scores."
    );
  });

  it("keeps first-run quality unknown without inventing weaknesses", () => {
    const snapshot = getInitialWorkspaceSnapshot();
    const quality = buildRetrievalQualityModel(
      buildRetrievalRuntimeModel(snapshot.retrieval, [])
    );

    expect(quality).toMatchObject({
      overallLevel: "unknown",
      overallLabel: "Retrieval quality unknown",
      overallDetail:
        "No selected retrieval evidence is available for a quality assessment.",
      hasEvidence: false,
      weaknesses: []
    });
    expect(quality.metrics.every((metric) => metric.level === "unknown")).toBe(
      true
    );
  });
});
