import React, { useState, useEffect, useRef } from "react";

export default function SqlDisplayPanel({
  useCoco,
  setUseCoco,
  sqlResult,
  setSearchResults,
  apiBase,
  setError,
}) {
  const [editableSQL, setEditableSQL] = useState("");
  const [draftSQL, setDraftSQL] = useState(editableSQL);
  const [isEditingSQL, setIsEditingSQL] = useState(false);
  const [queryRunLoading, setQueryRunLoading] = useState(false);
  const [totalCount, setTotalCount] = useState(null);
  const [countLoading, setCountLoading] = useState(false);
  const lastCountSqlRef = useRef("");

  const buildCountSql = (sql) => {
    const sqlWithoutLimit = sql
      .replace(/LIMIT\s+\d+/i, "")
      .replace(/;+$/, "")
      .trim();
    return `SELECT COUNT(*) as count FROM (${sqlWithoutLimit}) AS subquery`;
  };

  useEffect(() => {
    setTotalCount(null);
    setIsEditingSQL(false);
    setEditableSQL(sqlResult?.sql || "");
  }, [sqlResult]);

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
        const res = await fetch(`${apiBase}/api/run_query_count`, {
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

  const handleToggleEdit = () => {
    if (isEditingSQL) {
      setEditableSQL(draftSQL);
      setIsEditingSQL(false);
      return;
    }

    setDraftSQL(editableSQL);
    setIsEditingSQL(true);
  };

  const handleCancelEdit = () => {
    setDraftSQL(editableSQL);
    setIsEditingSQL(false);
  };

  const handleRunQuery = async () => {
    if (!editableSQL?.trim()) return;

    setQueryRunLoading(true);
    setError("");

    try {
      const res = await fetch(`${apiBase}/api/run_query`, {
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
    } catch (e) {
      setError(e?.message || "Run query failed");
    } finally {
      setQueryRunLoading(false);
    }
  };

  if (!sqlResult) return null;

  return (
    <div className="sql-panel">
      <div className="sql-stack">
        <div className="sql-card">
          <div className="sql-card-header">
            <div className="sql-card-title">SQL</div>
            <div className="sql-edit-actions">
              {isEditingSQL && (
                <button
                  onClick={handleCancelEdit}
                  className="sql-edit-btn"
                  title="Cancel"
                >
                  ❌ Cancel
                </button>
              )}
              <button
                onClick={handleToggleEdit}
                className="sql-edit-btn"
                title="Edit SQL"
              >
                {isEditingSQL ? "💾 Save" : "✏️ Edit"}
              </button>
            </div>
          </div>

          {isEditingSQL ? (
            <textarea
              value={draftSQL}
              onChange={(e) => setDraftSQL(e.target.value)}
              className="sql-textarea"
            />
          ) : (
            <pre className="sql-pre">{editableSQL}</pre>
          )}
        </div>

        {sqlResult.explanation && (
          <div className="reasoning-card">
            <div
              className="reasoning-title"
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>REASONING</span>
              <span
                className={
                  countLoading
                    ? "reasoning-count-badge is-loading"
                    : "reasoning-count-badge"
                }
              >
                {countLoading
                  ? "Counting total matches..."
                  : totalCount !== null
                    ? `Total Matches: ${totalCount}`
                    : "Total Matches: --"}
              </span>
            </div>
            <div className="reasoning-text">{sqlResult.explanation}</div>
          </div>
        )}
      </div>

      <button
        onClick={handleRunQuery}
        disabled={queryRunLoading}
        className="run-btn"
      >
        {queryRunLoading ? "Running…" : "▶ Run Query on Database"}
      </button>
    </div>
  );
}
