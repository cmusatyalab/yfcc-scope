import React from "react";

export default function SqlDisplayPanel({
  sqlResult,
  isEditingSQL,
  setIsEditingSQL,
  editableSQL,
  setEditableSQL,
  onRunQuery,
  queryRunLoading,
}) {
  if (!sqlResult) return null;

  return (
    <div className="sql-panel">
      <div className="sql-stack">
        <div className="sql-card">
          <div className="sql-card-header">
            <div className="sql-card-title">SQL</div>
            <button
              onClick={() => setIsEditingSQL(!isEditingSQL)}
              className="sql-edit-btn"
              title="Edit SQL"
            >
              ✏️ Edit
            </button>
          </div>

          {isEditingSQL ? (
            <textarea
              value={editableSQL}
              onChange={(e) => setEditableSQL(e.target.value)}
              className="sql-textarea"
            />
          ) : (
            <pre className="sql-pre">{editableSQL}</pre>
          )}
        </div>

        <div className="reasoning-card">
          <div className="reasoning-title">REASONING</div>
          <div className="reasoning-text">{sqlResult.explanation}</div>
        </div>
      </div>

      <button
        onClick={onRunQuery}
        disabled={queryRunLoading}
        className="run-btn"
      >
        {queryRunLoading ? "Running…" : "▶ Run Query on Database"}
      </button>
    </div>
  );
}
