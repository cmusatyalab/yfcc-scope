// Main image viewer component for YFCC viewer. User can enter their API key and use natural language to search for images. Renders a 3D first-person gallery of images returned from the search query.

import React, { useState, useEffect } from "react";
import Gallery, { makeLayout } from "./Gallery";
import SearchControlPanel from "./SearchControlPanel";
import SqlDisplayPanel from "./SqlDisplayPanel";
import ImageResultsPanel, { rowKey } from "./ImageResultsPanel";
import SelectedPanel from "./SelectedPanel";
import "./AppImageViewer.css";

// Read API base from env
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export default function App() {
  const [mode, setMode] = useState("search");
  const [useCoco, setUseCoco] = useState(true);
  const [sqlResult, setSqlResult] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [galleryItems, setGalleryItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const [showSelectedPanel, setShowSelectedPanel] = useState(false);

  useEffect(() => {
    setSelectedIds(new Set());
  }, [searchResults]);

  const handleEnterGallery = () => {
    if (!searchResults?.length) return;
    setGalleryItems(makeLayout(searchResults));
    setMode("gallery");
  };

  const toggleSelected = (row, index) => {
    const key = rowKey(row, index);
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const handleDownloadSelected = async () => {
    if (!searchResults?.length || selectedIds.size === 0) return;

    setDownloading(true);
    setError("");

    try {
      const idsToDownload = searchResults
        .filter((row, i) => selectedIds.has(rowKey(row, i)))
        .map((row) => row.image_file_id);

      // Create and submit a form to trigger the zip file download from Starlette API server
      const form = document.createElement("form");
      form.action = `${API_BASE}/api/download_zip`;
      form.method = "POST";
      form.style.display = "none";

      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "ids";
      input.value = JSON.stringify(idsToDownload);

      form.appendChild(input);
      document.body.appendChild(form);
      form.submit();
      form.remove();

      // UI state reset after a small delay since form.submit doesn't return a promise
      setTimeout(() => {
        setDownloading(false);
      }, 1500);
      return;
    } catch (e) {
      setError(e?.message || "Download failed");
      setDownloading(false);
    }
  };

  return (
    <div className="image-viewer-root">
      {mode === "gallery" ? (
        <Gallery
          items={galleryItems}
          onBack={() => setMode("search")}
          searchResults={searchResults}
          selectedIds={selectedIds}
          toggleSelected={toggleSelected}
          onDownloadSelected={handleDownloadSelected}
          downloading={downloading}
          showSelectedPanel={showSelectedPanel}
          setShowSelectedPanel={setShowSelectedPanel}
        />
      ) : (
        <div className="image-viewer-shell">
          <h1 className="app-title">3D Library Image Viewer</h1>
          {error && <p className="error-text">{error}</p>}

          <SearchControlPanel
            useCoco={useCoco}
            setUseCoco={setUseCoco}
            setSqlResult={setSqlResult}
            setSearchResults={setSearchResults}
            apiBase={API_BASE}
            setError={setError}
          />

          {useCoco && (
            <SqlDisplayPanel
              sqlResult={sqlResult}
              setSearchResults={setSearchResults}
              apiBase={API_BASE}
              setError={setError}
            />
          )}

          <ImageResultsPanel
            searchResults={searchResults}
            selectedIds={selectedIds}
            setSelectedIds={setSelectedIds}
            toggleSelected={toggleSelected}
            onDownloadSelected={handleDownloadSelected}
            downloading={downloading}
            onEnterGallery={handleEnterGallery}
            apiBase={API_BASE}
          />
        </div>
      )}

      <SelectedPanel
        showSelectedPanel={showSelectedPanel}
        setShowSelectedPanel={setShowSelectedPanel}
        searchResults={searchResults}
        selectedIds={selectedIds}
        toggleSelected={toggleSelected}
        onDownloadSelected={handleDownloadSelected}
        downloading={downloading}
      />
    </div>
  );
}
