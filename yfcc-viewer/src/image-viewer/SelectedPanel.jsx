import React from "react";
import { rowKey } from "./ImageResultsPanel";

export default function SelectedPanel({
  showSelectedPanel,
  setShowSelectedPanel,
  selectedIds,
  searchResults,
  toggleSelected,
  onDownloadSelected,
  downloading,
}) {
  if (showSelectedPanel) {
    return (
      <div
        className="selected-panel"
        onPointerDown={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="selected-panel-header">
          <div>Selected ({selectedIds.size})</div>

          <div className="selected-panel-actions">
            <button
              className="selected-panel-download"
              onClick={onDownloadSelected}
              disabled={selectedIds.size === 0 || downloading}
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
                <div className="selected-card-img-wrapper">
                  <img
                    src={row.thumb_url || row.path}
                    alt=""
                    className="selected-card-img"
                  />
                  <div className="selected-card-overlay">
                    <span>✖ Deselect</span>
                  </div>
                </div>
                <div className="selected-card-meta">{row.image_file_id}</div>
              </button>
            ))}
        </div>
      </div>
    );
  }

  if (selectedIds.size > 0) {
    return (
      <button
        className="floating-show-selected-btn"
        onClick={() => {
          if (document.exitPointerLock) {
            document.exitPointerLock();
          }
          setShowSelectedPanel(true);
        }}
      >
        Show Selected ({selectedIds.size})
      </button>
    );
  }

  return null;
}
