// Main image viewer component for YFCC viewer. User can enter their API key and use natural language to search for images. Renders a 3D first-person gallery of images returned from the search query.

import React, { useRef, useState, useEffect } from "react";
import buildSystemPrompt from "./sqlPrompt";
import "./AppImageViewer.css";

const API_BASE = "http://128.2.212.50:8081";

const LABELS = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic_light","fire_hydrant","stop_sign","parking_meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports_ball",
  "kite","baseball_bat","baseball_glove","skateboard","surfboard","tennis_racket",
  "bottle","wine_glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot_dog","pizza","donut","cake","chair",
  "couch","potted_plant","bed","dining_table","toilet","tv","laptop","mouse","remote",
  "keyboard","cell_phone","microwave","oven","toaster","sink","refrigerator","book",
  "clock","vase","scissors","teddy_bear","hair_drier","toothbrush",
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Search and ranking logic (Unused) ──────────────────────────────────────
function parseQuery(q) {
  return q
    .split(/[ ,]+/)
    .map((x) => x.trim().toLowerCase())
    .filter(Boolean)
    .filter((x, i, a) => a.indexOf(x) === i);
}
function buildVector(row) {
  return LABELS.map((l) => Number(row?.[l] || 0));
}
function cosineSim(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
function scoreRow(row, labels) {
  if (!labels.length) return Number(row?.total_bboxes || 0);
  let matched = 0,
    total = 0,
    best = 0;
  for (const l of labels) {
    const c = Number(row?.[l] || 0);
    if (c > 0) matched++;
    total += c;
    best = Math.max(best, c);
  }
  return (matched / labels.length) * 100 + total * 3 + best * 2;
}
function rankRows(rows, labels) {
  if (!rows.length) return [];
  const anchor = rows.reduce(
    (a, r) => (scoreRow(r, labels) > scoreRow(a, labels) ? r : a),
    rows[0],
  );
  const av = buildVector(anchor);
  return rows
    .map((r) => {
      const sim = cosineSim(av, buildVector(r));
      const qs = scoreRow(r, labels);
      const comb = labels.length
        ? sim * 0.72 + clamp(qs / 120, 0, 1) * 0.28
        : sim * 0.8 + clamp((r.total_bboxes || 0) / 20, 0, 1) * 0.2;
      return { ...r, combined: comb };
    })
    .sort((a, b) => b.combined - a.combined);
}

// ─── Layout constants ───────────────────────────────────────────────────────
const WALL_X = 440;
const IMG_W = 280;
const IMG_H = 195;
const MAT = 14;
const FRAME_W = IMG_W + MAT * 2;
const FRAME_H = IMG_H + MAT * 2;
const ROW_Y = FRAME_H / 2;

function makeLayout(items) {
  return items.map((item, i) => {
    const col = Math.floor(i / 4);
    const slot = i % 4;
    return {
      ...item,
      side: slot % 2 === 0 ? "left" : "right",
      wy: slot < 2 ? -ROW_Y : ROW_Y,
      wz: -(col * FRAME_W + 400),
    };
  });
}

function Gallery({ items, onBack }) {
  const divRef = useRef(null);
  const camRef = useRef({ x: 0, z: 0, yaw: 0 });
  const keysRef = useRef({});
  const rafRef = useRef();
  const [, tick] = useState(0);

  useEffect(() => {
    const SPD = 600;
    const ACCEL = 8;
    let velX = 0,
      velZ = 0;
    let last = performance.now();

    const loop = () => {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      const c = camRef.current;
      const k = keysRef.current;
      const sn = Math.sin(c.yaw),
        cs = Math.cos(c.yaw);
      let dx = 0,
        dz = 0;

      if (k["w"] || k["arrowup"]) {
        dx += sn;
        dz -= cs;
      }
      if (k["s"] || k["arrowdown"]) {
        dx -= sn;
        dz += cs;
      }
      if (k["d"] || k["arrowright"]) {
        dx += cs;
        dz += sn;
      }
      if (k["a"] || k["arrowleft"]) {
        dx -= cs;
        dz -= sn;
      }

      const len = Math.sqrt(dx * dx + dz * dz);
      if (len > 0) {
        dx /= len;
        dz /= len;
      }

      velX += (dx * SPD - velX) * ACCEL * dt;
      velZ += (dz * SPD - velZ) * ACCEL * dt;

      c.x = clamp(c.x + velX * dt, -(WALL_X - 100), WALL_X - 100);
      c.z = Math.min(100, c.z + velZ * dt);

      tick((n) => n + 1);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    const onKeyDown = (e) => {
      keysRef.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "q") onBack();
    };
    const onKeyUp = (e) => {
      keysRef.current[e.key.toLowerCase()] = false;
    };
    const onMove = (e) => {
      if (document.pointerLockElement !== divRef.current) return;
      camRef.current.yaw = clamp(
        camRef.current.yaw - e.movementX * 0.003,
        -1.25,
        1.25,
      );
    };
    const onClick = () => divRef.current?.requestPointerLock();

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    document.addEventListener("mousemove", onMove);
    divRef.current?.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      document.removeEventListener("mousemove", onMove);
    };
  }, [onBack]);

  const { x, z, yaw } = camRef.current;
  const worldTransform = `rotateY(${-yaw}rad) translate3d(${-x}px, 0px, ${-z}px)`;

  return (
    <div
      ref={divRef}
      className="gallery-root"
      style={{
        "--img-w": `${IMG_W}px`,
        "--img-h": `${IMG_H}px`,
        "--mat": `${MAT}px`,
      }}
    >
      <div className="gallery-world" style={{ transform: worldTransform }}>
        {(() => {
          const cols = Math.ceil(items.length / 4) || 1;
          const floorLen = cols * FRAME_W * 2 + 4000;
          return (
            <div
              className="gallery-floor"
              style={{
                width: WALL_X * 2,
                height: floorLen,
                left: -WALL_X,
                top: -floorLen / 2,
              }}
            />
          );
        })()}

        {items.map((item, i) => {
          const isLeft = item.side === "left";
          const wx = isLeft ? -(WALL_X - 6) : WALL_X - 6;
          const rotY = isLeft ? "90deg" : "-90deg";
          const src = item.thumb_url || item.path;

          return (
            <div
              key={i}
              className="gallery-frame-container"
              style={{
                transform: `translate3d(${wx}px, ${item.wy}px, ${item.wz}px) rotateY(${rotY})`,
              }}
            >
              <div className="gallery-frame">
                {src && <img src={src} alt="" className="gallery-img" />}
                <div className="gallery-caption">
                  {item.image_file_id || "—"}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="gallery-hud">
        Click to lock mouse · WASD to walk · Q to exit
      </div>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("cat dog");
  const [limit, setLimit] = useState(20);
  const [searchResults, setSearchResults] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("search");
  const [galleryItems, setGalleryItems] = useState([]);
  const [selectedIds, setSelectedIds] = useState(() => new Set());

  const [apiKey, setApiKey] = useState("");
  const [sqlLoading, setSqlLoading] = useState(false);
  const [sqlResult, setSqlResult] = useState(null);
  const [sqlError, setSqlError] = useState("");
  const [queryRunLoading, setQueryRunLoading] = useState(false);

  const [isEditingSQL, setIsEditingSQL] = useState(false);
  const [editableSQL, setEditableSQL] = useState("");

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
    } catch (e) {
      setError(e?.message || "Run query failed");
    } finally {
      setQueryRunLoading(false);
    }
  };

  const toggleSelected = (row, index) => {
    const key = row?.image_file_id ?? `idx-${index}`;
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

  if (mode === "gallery") {
    return <Gallery items={galleryItems} onBack={() => setMode("search")} />;
  }

  return (
    <div className="image-viewer-root">
      <div className="image-viewer-shell">
        <h1 className="app-title">3D Library Image Viewer</h1>

        <div className="input-row">
          <label htmlFor="apiKey" className="api-label">
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

        <div className="input-row">
          <label htmlFor="query" className="query-label">
            Image Search Query:
          </label>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleOpenAI();
            }}
            placeholder="two people kissing, a dog in a park…"
            className="query-input"
          />
        </div>

        <div className="input-row">
          <label htmlFor="limit" className="query-label">
            Number of images:
          </label>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="limit-select"
          >
            {[20, 40, 60, 80, 100, 150, 200, 300, 500].map((n) => (
              <option key={n} value={n}>
                {n} images
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleOpenAI}
          disabled={sqlLoading || !query.trim()}
          className="generate-btn"
        >
          {sqlLoading ? "Thinking…" : "Generate SQL"}
        </button>

        {sqlError && <p className="error-text">{sqlError}</p>}
        {error && <p className="error-text">{error}</p>}

        {sqlResult && (
          <>
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
                onClick={handleRunQuery}
                disabled={queryRunLoading}
                className="run-btn"
              >
                {queryRunLoading ? "Running…" : "▶ Run Query on Database"}
              </button>
            </div>
          </>
        )}

        {searchResults !== null && searchResults.length > 0 && (
          <div className="results-panel">
            <div className="results-header">
              <p className="results-count">{searchResults.length} results</p>
              <button onClick={handleEnter} className="enter-btn">
                Enter Gallery →
              </button>
            </div>

            <div className="results-list">
              {searchResults.map((row, i) => (
                <div
                  key={row.image_file_id || i}
                  className={`result-item ${selectedIds.has(row?.image_file_id ?? `idx-${i}`) ? "is-selected" : ""}`}
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
                    <img
                      src={row.thumb_url || row.path}
                      alt=""
                      className="result-img"
                    />

                    {/* IMAGE ID */}
                    <div className="result-meta">
                      <div>{row.image_file_id}</div>

                      <a
                        href={`http://128.2.212.50:8081/?image_file_id=${row.image_file_id}&select_all=1&min_conf=0.40`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="result-meta-link"
                      >
                        http://128.2.212.50:8081/?image_file_id=
                        {row.image_file_id}&select_all=1&min_conf=0.40
                      </a>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {searchResults !== null && searchResults.length === 0 && (
          <p className="empty-text">No images matched.</p>
        )}
      </div>
    </div>
  );
}
