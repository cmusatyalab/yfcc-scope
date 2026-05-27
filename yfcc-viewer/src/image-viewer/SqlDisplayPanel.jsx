import React, { useState } from "react";

export default function SqlDisplayPanel({
  sqlResult,
  isEditingSQL,
  setIsEditingSQL,
  editableSQL,
  setEditableSQL,
  onRunQuery,
  queryRunLoading,
  totalCount,
  countLoading,
}) {
  const [draftSQL, setDraftSQL] = useState(editableSQL);

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
