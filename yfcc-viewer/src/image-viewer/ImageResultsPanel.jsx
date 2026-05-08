import React from "react";

export function rowKey(row, index) {
  return row?.image_file_id ?? `idx-${index}`;
}

export default function ImageResultsPanel({
  searchResults,
  selectedIds,
  toggleSelected,
  onDownloadSelected,
  downloadLoading,
  downloadError,
  onEnterGallery,
  apiBase,
}) {
  if (searchResults === null) return null;

  if (searchResults.length === 0) {
    return <p className="empty-text">No images matched.</p>;
  }

  return (
    <>
      {downloadError && <p className="error-text">{downloadError}</p>}
      <div className="results-panel">
        <div className="results-header">
          <p className="results-count">{searchResults.length} results</p>
          <div className="results-actions">
            <button
              onClick={onDownloadSelected}
              className="download-btn"
              disabled={selectedIds.size === 0 || downloadLoading}
            >
              {downloadLoading
                ? "Preparing ZIP…"
                : `Download Selected (${selectedIds.size})`}
            </button>
            <button onClick={onEnterGallery} className="enter-btn">
              Enter Gallery →
            </button>
          </div>
        </div>

        <div className="results-list">
          {searchResults.map((row, i) => (
            <div
              key={row.image_file_id || i}
              className={`result-item ${selectedIds.has(rowKey(row, i)) ? "is-selected" : ""}`}
              onClick={() => toggleSelected(row, i)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  toggleSelected(row, i);
                }
              }}
            >
              <div className="result-card">
                <img
                  src={row.thumb_url || row.path}
                  alt=""
                  className="result-img"
                />

                <div className="result-meta">
                  <div>{row.image_file_id}</div>

                  <a
                    href={`${apiBase}/?image_file_id=${row.image_file_id}&select_all=1&min_conf=0.40`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="result-meta-link"
                  >
                    {apiBase}/?image_file_id=
                    {row.image_file_id}&select_all=1&min_conf=0.40
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
