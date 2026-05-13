import React from "react";

export default function SearchControlPanel({
  apiKey,
  setApiKey,
  query,
  setQuery,
  limit,
  setLimit,
  onGenerate,
  loading,
  error,
  sqlError,
}) {
  return (
    <>
      {error && <p className="error-text">{error}</p>}

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
          placeholder="a dog in a park"
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

      <button
        onClick={onGenerate}
        disabled={loading || !query.trim()}
        className="generate-btn"
      >
        {loading ? "Thinking…" : "Generate SQL"}
      </button>

      {sqlError && <p className="error-text">{sqlError}</p>}
    </>
  );
}
