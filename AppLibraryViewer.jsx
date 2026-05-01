import React, { useRef, useState, useEffect } from "react";

const API_BASE = "http://128.2.212.50:8080";

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

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function parseQuery(q) {
  return q.split(/[ ,]+/).map(x => x.trim().toLowerCase()).filter(Boolean)
    .filter((x, i, a) => a.indexOf(x) === i);
}
function buildVector(row) { return LABELS.map(l => Number(row?.[l] || 0)); }
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
function scoreRow(row, labels) {
  if (!labels.length) return Number(row?.total_bboxes || 0);
  let matched = 0, total = 0, best = 0;
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
  const anchor = rows.reduce((a, r) => scoreRow(r, labels) > scoreRow(a, labels) ? r : a, rows[0]);
  const av = buildVector(anchor);
  return rows.map(r => {
    const sim = cosineSim(av, buildVector(r));
    const qs = scoreRow(r, labels);
    const comb = labels.length
      ? sim * 0.72 + clamp(qs / 120, 0, 1) * 0.28
      : sim * 0.8 + clamp((r.total_bboxes || 0) / 20, 0, 1) * 0.2;
    return { ...r, combined: comb };
  }).sort((a, b) => b.combined - a.combined);
}

// ─── Layout constants ─────────────────────────────────────────────────────────
const WALL_X  = 440;
const IMG_W   = 280;
const IMG_H   = 195;
const MAT     = 14;
const FRAME_W = IMG_W + MAT * 2;
const FRAME_H = IMG_H + MAT * 2;
const ROW_Y   = FRAME_H / 2;

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
    let velX = 0, velZ = 0;
    let last = performance.now();

    const loop = () => {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      const c = camRef.current;
      const k = keysRef.current;
      const sn = Math.sin(c.yaw), cs = Math.cos(c.yaw);
      let dx = 0, dz = 0;

      if (k["w"] || k["arrowup"])    { dx += sn; dz -= cs; }
      if (k["s"] || k["arrowdown"])  { dx -= sn; dz += cs; }
      if (k["d"] || k["arrowright"]) { dx += cs; dz += sn; }
      if (k["a"] || k["arrowleft"])  { dx -= cs; dz -= sn; }

      const len = Math.sqrt(dx*dx + dz*dz);
      if (len > 0) { dx /= len; dz /= len; }

      velX += (dx * SPD - velX) * ACCEL * dt;
      velZ += (dz * SPD - velZ) * ACCEL * dt;

      c.x = clamp(c.x + velX * dt, -(WALL_X - 100), WALL_X - 100);
      c.z = Math.min(100, c.z + velZ * dt);

      tick(n => n + 1);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    const onKeyDown = (e) => {
      keysRef.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "q") onBack();
    };
    const onKeyUp = (e) => { keysRef.current[e.key.toLowerCase()] = false; };
    const onMove = (e) => {
      if (document.pointerLockElement !== divRef.current) return;
      camRef.current.yaw = clamp(
        camRef.current.yaw - e.movementX * 0.003,
        -1.25, 1.25
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
      style={{
        width: "100vw",
        height: "100vh",
        overflow: "hidden",
        background: "#000",
        perspective: "700px",
        perspectiveOrigin: "50% 50%",
        cursor: "crosshair",
        userSelect: "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: "50%",
          transformStyle: "preserve-3d",
          transform: worldTransform,
        }}
      >
        {(() => {
          const cols = Math.ceil(items.length / 4) || 1;
          const floorLen = cols * FRAME_W * 2 + 4000;
          return (
            <div
              style={{
                position: "absolute",
                width: WALL_X * 2,
                height: floorLen,
                left: -WALL_X,
                top: -floorLen / 2,
                background: "linear-gradient(to bottom, #c8c4be, #d8d4cf)",
                transformStyle: "preserve-3d",
                transform: "translateY(280px) rotateX(-90deg)",
              }}
            />
          );
        })()}

        {items.map((item, i) => {
          const isLeft = item.side === "left";
          const wx = isLeft ? -(WALL_X - 6) : (WALL_X - 6);
          const rotY = isLeft ? "90deg" : "-90deg";
          const src = item.thumb_url || item.path;

          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: -(IMG_W / 2 + MAT),
                top: -(IMG_H / 2 + MAT),
                transformStyle: "preserve-3d",
                transform: `translate3d(${wx}px, ${item.wy}px, ${item.wz}px) rotateY(${rotY})`,
              }}
            >
              <div
                style={{
                  position: "relative",
                  width: IMG_W + MAT * 2,
                  height: IMG_H + MAT * 2,
                  background: "#ffffff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
                }}
              >
                {src && (
                  <img
                    src={src}
                    alt=""
                    style={{
                      width: IMG_W,
                      height: IMG_H,
                      objectFit: "cover",
                      display: "block",
                    }}
                  />
                )}
                <div
                  style={{
                    position: "absolute",
                    bottom: 3,
                    left: 0,
                    right: 0,
                    fontSize: 9,
                    color: "#aaa",
                    fontFamily: "monospace",
                    textAlign: "center",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    padding: "0 6px",
                  }}
                >
                  {item.image_file_id || "—"}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 16,
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(0,0,0,0.45)",
          color: "rgba(255,255,255,0.65)",
          padding: "8px 20px",
          borderRadius: 999,
          fontSize: 12,
          backdropFilter: "blur(8px)",
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}
      >
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

  const buildSystemPrompt = (limitValue) => `You are a SQL expert. Given a user's natural language description of images they want to find, write a PostgreSQL SELECT query against these two tables:

Table: yfcc_index (image metadata and per-object counts)
  image_file_id TEXT PRIMARY KEY
  path TEXT
  ts TIMESTAMP
  total_bboxes INT
  person INT, bicycle INT, car INT, motorcycle INT, airplane INT, bus INT, train INT,
  truck INT, boat INT, traffic_light INT, fire_hydrant INT, stop_sign INT,
  parking_meter INT, bench INT, bird INT, cat INT, dog INT, horse INT, sheep INT,
  cow INT, elephant INT, bear INT, zebra INT, giraffe INT, backpack INT,
  umbrella INT, handbag INT, tie INT, suitcase INT, frisbee INT, skis INT,
  snowboard INT, sports_ball INT, kite INT, baseball_bat INT, baseball_glove INT,
  skateboard INT, surfboard INT, tennis_racket INT, bottle INT, wine_glass INT,
  cup INT, fork INT, knife INT, spoon INT, bowl INT, banana INT, apple INT,
  sandwich INT, orange INT, broccoli INT, carrot INT, hot_dog INT, pizza INT,
  donut INT, cake INT, chair INT, couch INT, potted_plant INT, bed INT,
  dining_table INT, toilet INT, tv INT, laptop INT, mouse INT, remote INT,
  keyboard INT, cell_phone INT, microwave INT, oven INT, toaster INT, sink INT,
  refrigerator INT, book INT, clock INT, vase INT, scissors INT, teddy_bear INT,
  hair_drier INT, toothbrush INT

Table: bb_table (individual bounding boxes)
  image_file_id TEXT REFERENCES yfcc_index(image_file_id)
  bounding_box_number INT
  label TEXT
  confidence_score FLOAT
  center_x FLOAT, center_y FLOAT, width FLOAT, height FLOAT

Rules:
- SELECT only image_file_id from yfcc_index. Nothing else.
- CRITICAL: You may ONLY filter on these exact label names: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign, parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket, bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed, dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_drier, toothbrush. Do not invent or use any other label names.
- If the user asks for something not directly in this list, map it creatively to the closest available labels and explain your mapping.
- Use yfcc_index count columns for simple presence/count filters (fast).
- Use bb_table when needed for ranking, spatial reasoning, object interaction, size, proximity, overlap, or relative position.
- If the query suggests interaction, touching, overlap, closeness, or relationships between objects, strongly prefer using bb_table geometry fields such as center_x, center_y, width, and height.
- When appropriate, reason about whether two boxes overlap or are close by comparing bounding box centers and sizes.
- When joining bb_table for ranking, restrict bb_table.label to the labels relevant to the query whenever possible so confidence ranking reflects the requested objects.
- ORDER results by relevance using confidence scores from bb_table.
- When using MAX() or AVG(), ALWAYS:
  - JOIN bb_table
  - GROUP BY yfcc_index.image_file_id

CRITICAL SPEED LIMITS:
- NEVER use correlated subqueries.
- NEVER use ORDER BY with a subquery.

BAD EXAMPLE (DO NOT DO THIS — TOO SLOW):
SELECT image_file_id
FROM yfcc_index
WHERE cat > 0 AND dog > 0
ORDER BY (
  SELECT AVG(confidence_score)
  FROM bb_table
  WHERE bb_table.image_file_id = yfcc_index.image_file_id
) DESC
LIMIT ${limitValue};

GOOD EXAMPLE (FAST AND CORRECT):
SELECT yfcc_index.image_file_id
FROM yfcc_index
JOIN bb_table ON yfcc_index.image_file_id = bb_table.image_file_id
WHERE yfcc_index.cat > 0 AND yfcc_index.dog > 0
  AND bb_table.label IN ('cat', 'dog')
GROUP BY yfcc_index.image_file_id
ORDER BY MAX(bb_table.confidence_score) DESC
LIMIT ${limitValue};

SPATIAL EXAMPLE (USE THIS STYLE WHEN INTERACTION/OVERLAP/CLOSENESS MATTERS):
SELECT yfcc_index.image_file_id
FROM yfcc_index
JOIN bb_table a
  ON yfcc_index.image_file_id = a.image_file_id
JOIN bb_table b
  ON a.image_file_id = b.image_file_id
 AND a.bounding_box_number < b.bounding_box_number
WHERE a.label = 'person'
  AND b.label = 'person'
  AND a.confidence_score >= 0.5
  AND b.confidence_score >= 0.5
  AND ABS(a.center_x - b.center_x) < (a.width + b.width) / 2
  AND ABS(a.center_y - b.center_y) < (a.height + b.height) / 2
GROUP BY yfcc_index.image_file_id
ORDER BY GREATEST(MAX(a.confidence_score), MAX(b.confidence_score)) DESC
LIMIT ${limitValue};

ADDITIONAL RULES:
- Prefer MAX(bb_table.confidence_score) for ranking (fast and stable).
- AVG(...) is allowed but only with JOIN + GROUP BY, never as a subquery.
- Avoid unnecessary nested queries, CTEs, or complex patterns if a simpler query works.
- For queries involving two instances of the same object or two different interacting objects, consider self-joining bb_table and using bounding_box_number to avoid duplicate pairs.
- For queries involving overlap, touching, holding, kissing, closeness, or adjacency, encourage use of center_x, center_y, width, and height if applicable.
- Always add LIMIT ${limitValue}.

OUTPUT FORMAT (STRICT JSON ONLY):
Return a valid JSON object with exactly these fields:
{
  "sql": string,
  "explanation": string
}

Rules for output:
- "sql" must contain ONLY the SQL query (no comments, no explanations).
- "explanation" must be exactly 2 sentences explaining:
  Sentence 1: Explain why you structured the SQL query this way (including joins, filters, ranking, and any use of bounding box geometry if applicable).
  Sentence 2: Evaluate whether LIMIT ${limitValue} is appropriate for this query to properly explore the possible results in the database, including whether this is enough to verify that these types of images actually exist and to assess the relevance, accuracy, and overall quality of the returned matches. Explicitly say whether the limit is too small, too large, or reasonable, and suggest a better number if needed.
`;

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
          "Authorization": `Bearer ${apiKey.trim()}`,
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
    } catch (e) {
      setError(e?.message || "Run query failed");
    } finally {
      setQueryRunLoading(false);
    }
  };

  if (mode === "gallery") {
    return <Gallery items={galleryItems} onBack={() => setMode("search")} />;
  }

  return (
    <div className="bg-[radial-gradient(circle_at_top,_#2a1d1a,_#120f0f_55%,_#090808)] text-white">
      <div className="px-4 pt-3 pb-6 w-full space-y-3">
        <input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          placeholder="sk-… API key"
          className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white outline-none placeholder:text-white/25 focus:border-white/25 w-48"
        />

        <h1 className="text-2xl font-bold tracking-tight text-white/90">3D Library Image Viewer</h1>

        <div className="flex gap-2 mt-4 w-1/2">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") handleOpenAI(); }}
            placeholder="two people kissing, a dog in a park…"
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm text-white outline-none placeholder:text-white/25 focus:border-white/25"
          />
          <select
            value={limit}
            onChange={e => setLimit(Number(e.target.value))}
            className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none"
          >
            {[20,40,60,80,100,150,200,300,500].map(n => (
              <option key={n} value={n}>{n} images</option>
            ))}
          </select>
          <button
            onClick={handleOpenAI}
            disabled={sqlLoading || !query.trim()}
            className="rounded-2xl border border-violet-400/40 bg-violet-400/10 px-5 py-3 text-sm font-semibold text-violet-200 transition hover:bg-violet-400/20 disabled:opacity-50 whitespace-nowrap"
          >
            {sqlLoading ? "Thinking…" : "Generate SQL"}
          </button>
        </div>

        {sqlError && <p className="text-sm text-red-300">{sqlError}</p>}
        {error && <p className="text-sm text-red-300">{error}</p>}

        {sqlResult && (
          <>
            <div style={{ width: "50%", marginBottom: "16px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div
                  style={{
                    background: "#1f1f1f",
                    border: "2px solid black",
                    borderRadius: "16px",
                    padding: "16px",
                    boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ fontSize: "11px", color: "#aaa", marginBottom: "8px" }}>
                      SQL
                    </div>
                    <button
                      onClick={() => setIsEditingSQL(!isEditingSQL)}
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "#aaa",
                        cursor: "pointer",
                        fontSize: "14px",
                        marginBottom: "8px",
                      }}
                      title="Edit SQL"
                    >
                      ✏️
                    </button>
                  </div>

                  {isEditingSQL ? (
                    <textarea
                      value={editableSQL}
                      onChange={(e) => setEditableSQL(e.target.value)}
                      style={{
                        width: "100%",
                        minHeight: "140px",
                        background: "#111",
                        color: "#4ade80",
                        border: "1px solid #333",
                        borderRadius: "8px",
                        padding: "10px",
                        fontSize: "12px",
                        fontFamily: "monospace",
                        resize: "vertical",
                        outline: "none",
                      }}
                    />
                  ) : (
                    <pre
                      style={{
                        color: "#4ade80",
                        fontSize: "12px",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {editableSQL}
                    </pre>
                  )}
                </div>

                <div
                  style={{
                    background: "#2a2a2a",
                    border: "2px solid black",
                    borderRadius: "16px",
                    padding: "16px",
                    boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                  }}
                >
                  <div style={{ fontSize: "11px", color: "#aaa", marginBottom: "8px" }}>
                    REASONING
                  </div>
                  <div style={{ color: "#ddd", fontSize: "14px", lineHeight: "1.5" }}>
                    {sqlResult.explanation}
                  </div>
                </div>
              </div>

              <button
                onClick={handleRunQuery}
                disabled={queryRunLoading}
                className="w-full mt-4 rounded-2xl border border-emerald-400/40 bg-emerald-400/10 px-6 py-3 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/20 disabled:opacity-50"
              >
                {queryRunLoading ? "Running…" : "▶ Run Query on Database"}
              </button>
            </div>
          </>
        )}

        {searchResults !== null && searchResults.length > 0 && (
          <div style={{ width: "50%" }}>
            <div className="flex items-center justify-between pt-1">
              <p className="text-xs text-white/40">{searchResults.length} results</p>
              <button
                onClick={handleEnter}
                className="rounded-full border border-amber-400/40 bg-amber-400/10 px-4 py-1.5 text-xs font-semibold text-amber-200 hover:bg-amber-400/20"
              >
                Enter Gallery →
              </button>
            </div>

            <div style={{ display: "flex", flexDirection: "column" }}>
              {searchResults.map((row, i) => (
                <div
                  key={row.image_file_id || i}
                  style={{ width: "400px", marginBottom: "12px" }}
                >
                  <img
                    src={row.thumb_url || row.path}
                    alt=""
                    style={{ width: "400px", display: "block" }}
                  />
        
                  {/* IMAGE ID */}
                  <div
                      style={{
                        fontSize: "11px",
                        color: "#aaa",
                        fontFamily: "monospace",
                        wordBreak: "break-all",
                        background: "#111",
                        padding: "4px 6px",
                        borderRadius: "6px",
                        marginTop: "4px",
                      }}
                    >
                      <div>{row.image_file_id}</div>
                    
                      <a
                        href={`http://128.2.212.50:8080/?image_file_id=${row.image_file_id}&select_all=1&min_conf=0.40`}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          color: "#60a5fa",
                          textDecoration: "underline",
                          fontSize: "10px",
                        }}
                      >
                        http://128.2.212.50:8080/?image_file_id={row.image_file_id}&select_all=1&min_conf=0.40
                      </a>
                    </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {searchResults !== null && searchResults.length === 0 && (
          <p className="text-sm text-white/35 py-6">No images matched.</p>
        )}
      </div>
    </div>
  );
}