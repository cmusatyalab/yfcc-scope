// Main image viewer component for YFCC viewer. User can enter their API key and use natural language to search for images. Renders a 3D first-person gallery of images returned from the search query.

import React, { useState, useEffect, useRef } from "react";
import buildSystemPrompt from "./sqlPrompt";
import Gallery, { makeLayout } from "./Gallery";
import SearchControlPanel from "./SearchControlPanel";
import SqlDisplayPanel from "./SqlDisplayPanel";
import ImageResultsPanel, { rowKey } from "./ImageResultsPanel";
import "./AppImageViewer.css";

// Read API base from env
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export default function App() {
  const [query, setQuery] = useState("cat dog");
  const [limit, setLimit] = useState(20);
  const [searchResults, setSearchResults] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("search");

  const [galleryItems, setGalleryItems] = useState([]);

  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [downloadError, setDownloadError] = useState("");

  const [apiKey, setApiKey] = useState("");
  const [sqlLoading, setSqlLoading] = useState(false);
  const [sqlResult, setSqlResult] = useState(null);
  const [sqlError, setSqlError] = useState("");
  const [queryRunLoading, setQueryRunLoading] = useState(false);

  const [isEditingSQL, setIsEditingSQL] = useState(false);
  const [editableSQL, setEditableSQL] = useState("");

  const [totalCount, setTotalCount] = useState(null);
  const [countLoading, setCountLoading] = useState(false);
  const lastCountSqlRef = useRef("");

  const handleEnter = () => {
    if (!searchResults?.length) return;
    setGalleryItems(makeLayout(searchResults));
    setMode("gallery");
  };

  const handleOpenAI = async () => {
    if (!query.trim()) return;
    if (!apiKey.trim()) {
      setSqlError("Enter an OpenAI API key first.");
      return;
    }

    setSqlLoading(true);
    setSqlResult(null);
    setSqlError("");
    setError("");
    setTotalCount(null);

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

      const parsed = JSON.parse(content);
      if (!parsed?.sql || !parsed?.explanation) {
        throw new Error("Structured response missing sql or explanation.");
      }

      setSqlResult(parsed);
      setEditableSQL(parsed.sql);
      setIsEditingSQL(false);
    } catch (e) {
      setSqlError(e?.message || "OpenAI request failed");
    } finally {
      setSqlLoading(false);
    }
  };

  const buildCountSql = (sql) => {
    const sqlWithoutLimit = sql
      .replace(/LIMIT\s+\d+/i, "")
      .replace(/;+$/, "")
      .trim();
    return `SELECT COUNT(*) as count FROM (${sqlWithoutLimit}) AS subquery`;
  };

  // Run COUNT(*) query whenever the saved SQL changes
  useEffect(() => {
    if (!editableSQL?.trim()) return;

    const countSql = buildCountSql(editableSQL);
    if (countSql === lastCountSqlRef.current) return;
    lastCountSqlRef.current = countSql;

    const controller = new AbortController();

    const fetchCount = async (sql, signal) => {
      setCountLoading(true);

      try {
        const res = await fetch(`${API_BASE}/api/run_query_count`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sql }),
          signal,
        });

        if (!res.ok) {
          throw new Error(`Server error (${res.status})`);
        }

        const data = await res.json();
        const rows = Array.isArray(data.rows) ? data.rows : [];
        if (signal.aborted) return;
        if (rows.length > 0) {
          setTotalCount(rows[0].count);
        } else {
          setTotalCount(0);
        }
      } catch (e) {
        if (e?.name === "AbortError") return;
        console.error("Count query failed", e);
        setTotalCount("Error");
      } finally {
        if (!signal.aborted) setCountLoading(false);
      }
    };

    fetchCount(countSql, controller.signal);

    return () => {
      controller.abort();
    };
  }, [editableSQL]);

  const handleRunQuery = async () => {
    if (!editableSQL?.trim()) return;

    setQueryRunLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/api/run_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sql: editableSQL }),
      });

      if (!res.ok) {
        const t = await res.json();
        throw new Error(t.error || `Server error (${res.status})`);
      }

      const data = await res.json();
      const rows = Array.isArray(data.rows) ? data.rows : [];

      if (rows.length === 0) {
        setSearchResults([]);
        setError("Query ran but returned 0 results.");
        return;
      }

      setSearchResults(rows);
      setSelectedIds(new Set());
      setDownloadError("");
    } catch (e) {
      setError(e?.message || "Run query failed");
    } finally {
      setQueryRunLoading(false);
    }
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

  const handleSelectAll = () => {
    if (!searchResults) return;
    setSelectedIds(new Set(searchResults.map((row, i) => rowKey(row, i))));
  };

  const handleClearSelection = () => {
    setSelectedIds(new Set());
  };

  const handleDownloadSelected = async () => {
    if (!searchResults?.length || selectedIds.size === 0) return;

    setDownloadLoading(true);
    setDownloadError("");

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
        setDownloadLoading(false);
      }, 1500);
      return;
    } catch (e) {
      setDownloadError(e?.message || "Download failed");
      setDownloadLoading(false);
    }
  };

  if (mode === "gallery") {
    return (
      <Gallery
        items={galleryItems}
        onBack={() => setMode("search")}
        searchResults={searchResults}
        selectedIds={selectedIds}
        toggleSelected={toggleSelected}
        onDownloadSelected={handleDownloadSelected}
        downloadLoading={downloadLoading}
      />
    );
  }

  return (
    <div className="image-viewer-root">
      <div className="image-viewer-shell">
        <h1 className="app-title">3D Library Image Viewer</h1>

        <SearchControlPanel
          apiKey={apiKey}
          setApiKey={setApiKey}
          query={query}
          setQuery={setQuery}
          limit={limit}
          setLimit={setLimit}
          onGenerate={handleOpenAI}
          loading={sqlLoading}
          error={error}
          sqlError={sqlError}
        />

        <SqlDisplayPanel
          sqlResult={sqlResult}
          isEditingSQL={isEditingSQL}
          setIsEditingSQL={setIsEditingSQL}
          editableSQL={editableSQL}
          setEditableSQL={setEditableSQL}
          onRunQuery={handleRunQuery}
          queryRunLoading={queryRunLoading}
          totalCount={totalCount}
          countLoading={countLoading}
        />

        <ImageResultsPanel
          searchResults={searchResults}
          selectedIds={selectedIds}
          toggleSelected={toggleSelected}
          onSelectAll={handleSelectAll}
          onClearSelection={handleClearSelection}
          onDownloadSelected={handleDownloadSelected}
          downloadLoading={downloadLoading}
          downloadError={downloadError}
          onEnterGallery={handleEnter}
          apiBase={API_BASE}
        />
      </div>
    </div>
  );
}
