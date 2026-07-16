import React, { useState } from "react";

export function rowKey(row, index) {
  return row?.image_file_id ?? `idx-${index}`;
}

export default function ImageResultsPanel({
  searchResults,
  selectedIds,
  setSelectedIds,
  toggleSelected,
  onDownloadSelected,
  downloading,
  onEnterGallery,
  apiBase,
}) {
  const baseUrl = apiBase || window.location.origin;
  const [enlargedImage, setEnlargedImage] = useState(null);
  const [showSelectedPanel, setShowSelectedPanel] = useState(false);

  const handleSelectAll = () => {
    if (!searchResults) return;
    setSelectedIds(new Set(searchResults.map((row, i) => rowKey(row, i))));
  };

  const handleClearSelection = () => {
    setSelectedIds(new Set());
  };

  if (searchResults === null) return null;

  if (searchResults.length === 0) {
    return <p className="empty-text">No images matched.</p>;
  }

  return (
    <>
      <div className="results-panel">
        <div className="results-header">
          <div className="results-count">{searchResults.length} results</div>
          <div className="results-actions">
            <button onClick={handleSelectAll} className="select-btn">
              Select All
            </button>

            <button
              onClick={handleClearSelection}
              className="select-btn"
              disabled={selectedIds.size === 0}
            >
              Clear Selection
            </button>

            <button
              onClick={onDownloadSelected}
              className="enter-btn"
              disabled={selectedIds.size === 0 || downloading}
            >
              {downloading
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
                    href={`${baseUrl}/boxviewer?image_file_id=${row.image_file_id}&select_all=1&min_conf=0.40`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="result-meta-link"
                  >
                    {baseUrl}/boxviewer?image_file_id=
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
    </>
  );
}
