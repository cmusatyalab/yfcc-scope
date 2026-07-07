import React, { useState } from "react";
import { getErrorMessage } from "./utils";
import buildSystemPrompt from "./sqlPrompt";

export default function CocoSearchPanel({
  query,
  setQuery,
  limit,
  setSqlResult,
  setError,
}) {
  const [apiKey, setApiKey] = useState("");
  const [reqLoading, setReqLoading] = useState(false);

  const handleOpenAI = async () => {
    if (!query.trim()) return;
    if (!apiKey.trim()) {
      setError("Enter an OpenAI API key first.");
      return;
    }

    setReqLoading(true);
    setSqlResult(null);
    setError("");

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
        throw new Error(await getErrorMessage(res));
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
      setReqLoading(false);
    }
  };

  const isSearchDisabled = reqLoading || !query.trim();

  return (
    <>
      <div className="input-row">
        <label htmlFor="query" className="query-label">
          Image Search Query:
        </label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleOpenAI();
          }}
          placeholder="e.g. red tailed hawk"
          className="query-input"
        />
      </div>

      <div className="input-row">
        <label className="api-label">OpenAI API Key:</label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-xxxxxxxx"
          className="api-input"
        />
      </div>

      <button
        onClick={handleOpenAI}
        disabled={isSearchDisabled}
        className="generate-btn"
      >
        {reqLoading ? "Thinking…" : "Generate SQL"}
      </button>
    </>
  );
}
