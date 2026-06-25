import React, { useState, useEffect, useRef, useCallback } from "react";
import buildSystemPrompt from "./sqlPrompt";

export default function SearchControlPanel({
  useCoco,
  setUseCoco,
  setSqlResult,
  setSearchResults,
  apiBase,
  setError,
}) {
  const [query, setQuery] = useState("hawk");
  const [apiKey, setApiKey] = useState("");
  const [limit, setLimit] = useState(20);
  const [reqLoading, setReqLoading] = useState(false);
  const [searchInput, setSearchInput] = useState("text");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  const onGenerate = async () => {
    if (useCoco) {
      handleOpenAI();
    } else if (searchInput === "image") {
      handleClipImage();
    } else {
      handleClip();
    }
  };

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
        const t = await res.text();
        throw new Error(t);
      }

      const data = await res.json();
      const content = data.choices?.[0]?.message?.content?.trim();
      if (!content) throw new Error("No structured response returned.");

      const parsed = JSON.parse(content); // {sql: "...", explanation: "..."}
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
        const t = await res.json();
        throw new Error(t.error || `Server error (${res.status})`);
      }

      const data = await res.json(); // {embedding: [...], rows: [...]}
      const embedding = data?.embedding;
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
        const t = await res.json();
        throw new Error(t.error || `Server error (${res.status})`);
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

  const handleFilePick = useCallback((e) => {
    showFile(e.target.files[0]);
  }, [showFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    if (dropZoneRef.current) dropZoneRef.current.classList.add("drag-over");
  }, []);

  const handleDragLeave = useCallback(() => {
    if (dropZoneRef.current) dropZoneRef.current.classList.remove("drag-over");
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    if (dropZoneRef.current) dropZoneRef.current.classList.remove("drag-over");
    showFile(e.dataTransfer.files[0]);
  }, [showFile]);

  useEffect(() => {
    if (searchInput !== "image") return;
    const handler = (e) => {
      for (const item of (e.clipboardData?.items || [])) {
        if (item.type.startsWith("image/")) {
          showFile(item.getAsFile());
          break;
        }
      }
    };
    document.addEventListener("paste", handler);
    return () => document.removeEventListener("paste", handler);
  }, [searchInput, showFile]);

  const isSearchDisabled = reqLoading || (searchInput === "image" ? !imageFile : !query.trim());

  return (
    <>
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

      {!useCoco && (
        <div className="input-row">
          <label className="query-label">
            Search Input:
          </label>
          <div className="toggle-pill">
            <button
              className={"toggle-btn" + (searchInput === "text" ? " active" : "")}
              onClick={() => setSearchInput("text")}
            >
              Text query
            </button>
            <button
              className={"toggle-btn" + (searchInput === "image" ? " active" : "")}
              onClick={() => setSearchInput("image")}
            >
              Image upload
            </button>
          </div>
        </div>
      )}

      {(useCoco || (!useCoco && searchInput === "text")) && (
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

      {!useCoco && searchInput === "image" && (
        <div className="input-row">
          <label className="query-label">
            Upload Image:
          </label>
          <div>
            <div
              className={"upload-zone" + (imagePreview ? " hidden" : "")}
              ref={dropZoneRef}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <p className="upload-title">Drag & drop / Paste from clipboard / Click to browse</p>
              <p className="upload-sub">Supported formats: PNG, JPG, WebP</p>
            </div>

            <div className={"preview-area" + (imagePreview ? " visible" : "")}>
              <img className="preview-thumb" src={imagePreview || ""} alt="preview" />
              <div className="preview-meta">
                <p className="preview-name">{imageFile?.name || ""}</p>
                <p className="preview-size">{imageFile ? (imageFile.size / 1024).toFixed(0) + " KB · " + imageFile.type : ""}</p>
              </div>
              <button className="preview-clear" onClick={clearImage}>
                Clear
              </button>
            </div>

            <input type="file" ref={fileInputRef} accept="image/*" style={{ display: "none" }} onChange={handleFilePick} />
          </div>
        </div>
      )}

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

      {useCoco && (
        <div className="input-row">
          <label className="api-label">
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
        disabled={isSearchDisabled}
        className="generate-btn"
      >
        {useCoco
          ? reqLoading
            ? "Thinking…"
            : "Generate SQL"
          : reqLoading
            ? "Searching…"
            : "Search nearest neighbor images"}
      </button>
    </>
  );
}
