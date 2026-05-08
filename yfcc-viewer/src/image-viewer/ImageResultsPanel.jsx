import React, { useState } from "react";

export function rowKey(row, index) {
  return row?.image_file_id ?? `idx-${index}`;
}

export default function ImageResultsPanel({
  searchResults,
  selectedIds,
  toggleSelected,
  onSelectAll,
  onClearSelection,
  onDownloadSelected,
  downloadLoading,
  downloadError,
  onEnterGallery,
  apiBase,
}) {
  const [enlargedImage, setEnlargedImage] = useState(null);
  const [showSelectedPanel, setShowSelectedPanel] = useState(false);

  if (searchResults === null) return null;

  if (searchResults.length === 0) {
    return <p className="empty-text">No images matched.</p>;
  }

  return (
    <>
      {downloadError && <p className="error-text">{downloadError}</p>}
      <div className="results-panel">
        <div className="results-header">
          <div className="results-count">{searchResults.length} results</div>
          <div className="results-actions">
            <button onClick={onSelectAll} className="select-btn">
              Select All
            </button>

            <button
              onClick={onClearSelection}
              className="select-btn"
              disabled={selectedIds.size === 0}
            >
              Clear Selection
            </button>

            <button
              onClick={() => setShowSelectedPanel((prev) => !prev)}
              className="select-btn"
              disabled={selectedIds.size === 0}
            >
              {showSelectedPanel
                ? "Hide Selected"
                : `Show Selected (${selectedIds.size})`}
            </button>

            <button
              onClick={onDownloadSelected}
              className="enter-btn"
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
                <div style={{ position: "relative" }} className="img-container">
                  <img
                    src={row.thumb_url || row.path}
                    alt=""
                    className="result-img"
                  />
                  <button
                    className="enlarge-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setEnlargedImage(row.path);
                    }}
                    title="Enlarge Image"
                  >
                    🔍
                  </button>
                </div>

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

        {enlargedImage && (
          <div
            className="image-enlarge-overlay"
            onClick={() => setEnlargedImage(null)}
          >
            <img
              src={enlargedImage}
              alt="Enlarged"
              className="image-enlarge-content"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              className="image-enlarge-close"
              onClick={() => setEnlargedImage(null)}
            >
              ✖
            </button>
          </div>
        )}
      </div>

      {showSelectedPanel && (
        <div className="selected-panel">
          <div className="selected-panel-header">
            <div>Selected ({selectedIds.size})</div>

            <div className="selected-panel-actions">
              <button
                className="selected-panel-download"
                onClick={onDownloadSelected}
                disabled={selectedIds.size === 0 || downloadLoading}
                aria-label="Download selected"
                title="Download selected"
              >
                ⬇
              </button>
              <button
                className="selected-panel-close"
                onClick={() => setShowSelectedPanel(false)}
                aria-label="Close selected panel"
              >
                ✖
              </button>
            </div>
          </div>

          <div className="selected-panel-grid">
            {searchResults
              .filter((row, i) => selectedIds.has(rowKey(row, i)))
              .map((row, i) => (
                <button
                  key={`selected-${rowKey(row, i)}`}
                  className="selected-card"
                  onClick={() => toggleSelected(row, i)}
                  title="Click to deselect"
                >
                  <img
                    src={row.thumb_url || row.path}
                    alt=""
                    className="selected-card-img"
                  />
                  <div className="selected-card-meta">{row.image_file_id}</div>
                </button>
              ))}
          </div>
        </div>
      )}
    </>
  );
}
