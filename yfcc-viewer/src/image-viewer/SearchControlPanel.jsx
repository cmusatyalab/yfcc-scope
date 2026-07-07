import React, { useState } from "react";
import CocoSearchPanel from "./CocoSearchPanel";
import ClipSearchPanel from "./ClipSearchPanel";

export default function SearchControlPanel({
  query,
  setQuery,
  useCoco,
  setUseCoco,
  setSqlResult,
  setSearchResults,
  apiBase,
  setError,
}) {
  const [limit, setLimit] = useState(20);

  return (
    <div className="search-control-panel">
      <div className="input-row">
        <label className="query-label">
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
        <label className="query-label">
          Number of images:
        </label>
        <select
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="limit-select"
        >
          {[20, 40, 60, 80, 100, 150, 200, 300, 400, 500].map((n) => (
            <option key={n} value={n}>
              {n} images
            </option>
          ))}
        </select>
      </div>

      {useCoco ? (
        <CocoSearchPanel
          query={query}
          setQuery={setQuery}
          limit={limit}
          setSqlResult={setSqlResult}
          setError={setError}
        />
      ) : (
        <ClipSearchPanel
          query={query}
          setQuery={setQuery}
          limit={limit}
          apiBase={apiBase}
          setSearchResults={setSearchResults}
          setError={setError}
        />
      )}
    </div>
  );
}
