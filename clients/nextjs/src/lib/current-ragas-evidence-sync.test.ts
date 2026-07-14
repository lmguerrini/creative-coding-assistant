import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  CURRENT_CANONICAL_RETRIEVAL_REPORT,
  CURRENT_PRODUCT_RAGAS_EVIDENCE
} from "./current-ragas-evidence";
import {
  currentProductRetrievalScoreFromEvidence,
  type RagasCaseEvidence,
  type RagasExecutionEvidence
} from "./evaluation-benchmark";

type CanonicalCaseResult = Omit<RagasCaseEvidence, "sampleId"> & {
  caseId: string;
};

type CanonicalEvidenceFile = Omit<
  RagasExecutionEvidence,
  "state" | "caseRows"
> & {
  status: RagasExecutionEvidence["state"];
  caseResults: CanonicalCaseResult[];
};

const canonicalEvidencePath = resolve(
  process.cwd(),
  "../../demo/evaluation/current_product_ragas_evidence.json"
);
const canonicalRetrievalReportPath = resolve(
  process.cwd(),
  "../../demo/evaluation/canonical_retrieval_report.json"
);

function projectCanonicalEvidence(
  payload: CanonicalEvidenceFile
): RagasExecutionEvidence {
  const { status, caseResults, ...evidence } = payload;
  return {
    ...evidence,
    state: status,
    caseRows: caseResults.map(({ caseId, ...row }) => ({
      ...row,
      sampleId: caseId
    }))
  };
}

describe("committed current-product evidence", () => {
  it("keeps the Dashboard retrieval coverage synchronized with its report", () => {
    const report = JSON.parse(
      readFileSync(canonicalRetrievalReportPath, "utf8")
    ) as {
      benchmarkCaseCount: number;
      evaluatedAt: string;
      selectionFingerprint: string;
      summary: {
        casesWithResults: number;
        expectedSourceOverlap: { covered: number; expected: number; ratio: number };
        requestedDomainCoverage: { covered: number; expected: number; ratio: number };
      };
    };

    expect(CURRENT_CANONICAL_RETRIEVAL_REPORT).toMatchObject({
      evaluatedAt: report.evaluatedAt,
      selectionFingerprint: report.selectionFingerprint,
      retrievalPackCases: report.benchmarkCaseCount,
      casesWithResults: report.summary.casesWithResults,
      expectedSourceOverlap: report.summary.expectedSourceOverlap,
      requestedDomainCoverage: report.summary.requestedDomainCoverage
    });
  });

  it("keeps the Dashboard static projection synchronized with canonical JSON", () => {
    if (!existsSync(canonicalEvidencePath)) {
      expect(CURRENT_PRODUCT_RAGAS_EVIDENCE).toBeNull();
      return;
    }

    const canonical = JSON.parse(
      readFileSync(canonicalEvidencePath, "utf8")
    ) as CanonicalEvidenceFile;
    const projected = projectCanonicalEvidence(canonical);

    expect(CURRENT_PRODUCT_RAGAS_EVIDENCE).toEqual(projected);
    expect(currentProductRetrievalScoreFromEvidence(projected)).toBe(
      canonical.retrievalScore
    );
  });
});
