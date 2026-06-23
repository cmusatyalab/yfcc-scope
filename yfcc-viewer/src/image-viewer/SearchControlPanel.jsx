import React, { useState, useEffect } from "react";
import buildSystemPrompt from "./sqlPrompt";

export default function SearchControlPanel({
  useCoco,
  setUseCoco,
  setSqlResult,
  setError,
}) {
  const [query, setQuery] = useState("hawk");
  const [apiKey, setApiKey] = useState("");
  const [limit, setLimit] = useState(20);
  const [sqlLoading, setSqlLoading] = useState(false);

  const onGenerate = async () => {
    if (!query.trim()) return;
    if (useCoco) {
      handleOpenAI();
    } else {
      // TODO: search clip embedding nearest neighbors
      setError("");
    }
  };

  const handleOpenAI = async () => {
    if (!query.trim()) return;
    if (!apiKey.trim()) {
      setError("Enter an OpenAI API key first.");
      return;
    }

    setSqlLoading(true);
    setSqlResult(null);
    setError("");
    // setError("");

    try {
      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey.trim()}`,
        },
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [
            { role: "system", content: buildSystemPrompt(limit) },
            { role: "user", content: query },
          ],
          temperature: 0,
          response_format: {
            type: "json_schema",
            json_schema: {
              name: "sql_query_response",
              strict: true,
              schema: {
                type: "object",
                additionalProperties: false,
                properties: {
                  sql: { type: "string" },
                  explanation: { type: "string" },
                },
                required: ["sql", "explanation"],
              },
            },
          },
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t);
      }

      const data = await res.json();
      const content = data.choices?.[0]?.message?.content?.trim();
      if (!content) throw new Error("No structured response returned.");

      const parsed = JSON.parse(content);
      if (!parsed?.sql || !parsed?.explanation) {
        throw new Error("Structured response missing sql or explanation.");
      }
      setSqlResult(parsed);
    } catch (e) {
      setError(e?.message || "OpenAI request failed");
    } finally {
      setSqlLoading(false);
    }
  };

  const handleClip = () => {};

  return (
    <>
      <div className="input-row">
        <label htmlFor="limit" className="query-label">
          Search Using:
        </label>
        <select
          value={useCoco ? "coco" : "clip"}
          onChange={(e) => setUseCoco(e.target.value === "coco")}
          className="limit-select"
        >
          <option value="coco">YOLO Detected COCO Classes</option>
          <option value="clip">CLIP Embedding</option>
        </select>
      </div>

      <div className="input-row">
        <label htmlFor="query" className="query-label">
          Image Search Query:
        </label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") onGenerate();
          }}
          placeholder="hawk"
          className="query-input"
        />
      </div>

      <div className="input-row">
        <label htmlFor="limit" className="query-label">
          Number of images:
        </label>
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="limit-select"
        >
          {[20, 40, 60, 80, 100, 150].map((n) => (
            <option key={n} value={n}>
              {n} images
            </option>
          ))}
        </select>
      </div>

      {useCoco && (
        <div className="input-row">
          <label htmlFor="apiKey" className="api-label">
            OpenAI API Key:
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-xxxxxxxx"
            className="api-input"
          />
        </div>
      )}

      <button
        onClick={onGenerate}
        disabled={sqlLoading || !query.trim()}
        className="generate-btn"
      >
        {useCoco
          ? sqlLoading
            ? "Thinking…"
            : "Generate SQL"
          : "Search nearest neighbor images"}
      </button>
    </>
  );
}
