import React, { useState, useEffect, useRef, useCallback } from "react";
import { getErrorMessage } from "./utils";

export default function ClipSearchPanel({
  query,
  setQuery,
  limit,
  apiBase,
  setSearchResults,
  setError,
}) {
  const [inputType, setInputType] = useState("text");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [reqLoading, setReqLoading] = useState(false);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  const handleClip = async () => {
    if (!query.trim()) return;

    setReqLoading(true);
    setSearchResults(null);
    setError("");

    try {
      const res = await fetch(`${apiBase}/api/clip_text_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: query.trim(), limit: limit }),
      });

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      const data = await res.json();
      const rows = data?.rows;

      if (rows.length === 0) {
        setSearchResults([]);
        setError("Query ran but returned 0 results.");
        return;
      }
      setSearchResults(rows);
    } catch (e) {
      setError(e?.message || "Run query failed");
    } finally {
      setReqLoading(false);
    }
  };

  const handleClipImage = async () => {
    if (!imageFile) return;

    setReqLoading(true);
    setSearchResults(null);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("limit", String(limit));

      const res = await fetch(`${apiBase}/api/clip_image_query`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await getErrorMessage(res));
      }

      const data = await res.json();
      const rows = data?.rows;

      if (rows.length === 0) {
        setSearchResults([]);
        setError("Query ran but returned 0 results.");
        return;
      }
      setSearchResults(rows);
    } catch (e) {
      setError(e?.message || "Image query failed");
    } finally {
      setReqLoading(false);
    }
  };

  const showFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }, []);

  const clearImage = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [imagePreview]);

  const handleFilePick = useCallback(
    (e) => {
      showFile(e.target.files[0]);
    },
    [showFile],
  );

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    if (dropZoneRef.current) dropZoneRef.current.classList.add("drag-over");
  }, []);

  const handleDragLeave = useCallback(() => {
    if (dropZoneRef.current) dropZoneRef.current.classList.remove("drag-over");
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      if (dropZoneRef.current)
        dropZoneRef.current.classList.remove("drag-over");
      showFile(e.dataTransfer.files[0]);
    },
    [showFile],
  );

  useEffect(() => {
    if (inputType !== "image") return;
    const handler = (e) => {
      for (const item of e.clipboardData?.items || []) {
        if (item.type.startsWith("image/")) {
          showFile(item.getAsFile());
          break;
        }
      }
    };
    document.addEventListener("paste", handler);
    return () => document.removeEventListener("paste", handler);
  }, [inputType, showFile]);

  const isSearchDisabled =
    reqLoading || (inputType === "image" ? !imageFile : !query.trim());
  const onGenerate = () =>
    inputType === "image" ? handleClipImage() : handleClip();

  return (
    <>
      <div className="input-row">
        <label className="query-label">Search Input:</label>
        <div className="toggle-pill">
          <button
            className={"toggle-btn" + (inputType === "text" ? " active" : "")}
            onClick={() => setInputType("text")}
          >
            Text query
          </button>
          <button
            className={"toggle-btn" + (inputType === "image" ? " active" : "")}
            onClick={() => setInputType("image")}
          >
            Image upload
          </button>
        </div>
      </div>

      {inputType === "text" && (
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
            placeholder="e.g. red tailed hawk"
            className="query-input"
          />
        </div>
      )}

      {inputType === "image" && (
        <div className="input-row">
          <label className="query-label">Upload Image:</label>
          <div>
            <div
              className={"upload-zone" + (imagePreview ? " hidden" : "")}
              ref={dropZoneRef}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <p className="upload-title">
                Drag & drop / Paste from clipboard / Click to browse
              </p>
              <p className="upload-sub">Supported formats: PNG, JPG, WebP</p>
            </div>

            <div className={"preview-area" + (imagePreview ? " visible" : "")}>
              <img
                className="preview-thumb"
                src={imagePreview || ""}
                alt="preview"
              />
              <div className="preview-meta">
                <p className="preview-name">{imageFile?.name || ""}</p>
                <p className="preview-size">
                  {imageFile
                    ? (imageFile.size / 1024).toFixed(0) +
                      " KB · " +
                      imageFile.type
                    : ""}
                </p>
              </div>
              <button className="preview-clear" onClick={clearImage}>
                Clear
              </button>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFilePick}
            />
          </div>
        </div>
      )}

      <button
        onClick={onGenerate}
        disabled={isSearchDisabled}
        className="generate-btn"
      >
        {reqLoading ? "Searching…" : "Search nearest neighbor images"}
      </button>
    </>
  );
}
