"use client";

import { useState } from "react";
import type { FeedbackSentiment } from "@/lib/product-controls";

export function OutputFeedbackPanel({
  artifactTitle,
  onSubmit
}: {
  artifactTitle: string;
  onSubmit: (sentiment: FeedbackSentiment, comment: string | null) => void;
}) {
  const [comment, setComment] = useState("");
  const [submitted, setSubmitted] = useState<FeedbackSentiment | null>(null);

  function submit(sentiment: FeedbackSentiment) {
    onSubmit(sentiment, comment);
    setComment("");
    setSubmitted(sentiment);
  }

  return (
    <section className="outputFeedback" aria-label="Output feedback">
      <div>
        <strong>How did {artifactTitle} work for you?</strong>
        <span>Feedback is stored locally as an explicit preference signal.</span>
      </div>
      <label>
        <span className="srOnly">Optional feedback comment</span>
        <input
          onChange={(event) => setComment(event.currentTarget.value)}
          placeholder="Optional note"
          value={comment}
        />
      </label>
      <div className="outputFeedbackActions">
        <button
          aria-label="Mark output helpful"
          aria-pressed={submitted === "positive"}
          onClick={() => submit("positive")}
          type="button"
        >
          Helpful
        </button>
        <button
          aria-label="Mark output not helpful"
          aria-pressed={submitted === "negative"}
          onClick={() => submit("negative")}
          type="button"
        >
          Needs work
        </button>
      </div>
    </section>
  );
}
