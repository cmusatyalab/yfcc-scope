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
  for (const l of labels) { const c = Number(row?.[l]||0); if(c>0)matched++; total+=c; best=Math.max(best,c); }
  return (matched / labels.length) * 100 + total * 3 + best * 2;
}
function rankRows(rows, labels) {
  if (!rows.length) return [];
  const anchor = rows.reduce((a,r) => scoreRow(r,labels)>scoreRow(a,labels)?r:a, rows[0]);
  const av = buildVector(anchor);
  return rows.map(r => {
    const sim  = cosineSim(av, buildVector(r));
    const qs   = scoreRow(r, labels);
    const comb = labels.length
      ? sim*0.72 + clamp(qs/120,0,1)*0.28
      : sim*0.8  + clamp((r.total_bboxes||0)/20,0,1)*0.2;
    return {...r, combined: comb};
  }).sort((a,b) => b.combined - a.combined);
}

// ─── Layout constants ─────────────────────────────────────────────────────────
const WALL_X  = 440;   // half-corridor width (px)
const IMG_W   = 280;
const IMG_H   = 195;
const MAT     = 14;    // white mat border (px)
const FRAME_W = IMG_W + MAT * 2;   // 308 px — column pitch
const FRAME_H = IMG_H + MAT * 2;   // 223 px — row height
const ROW_Y   = FRAME_H / 2;       // vertical offset for top/bottom rows

function makeLayout(items) {
  return items.map((item, i) => {
    const col  = Math.floor(i / 4);
    const slot = i % 4;   // 0=left-top 1=right-top 2=left-bottom 3=right-bottom
    return {
      ...item,
      side: slot % 2 === 0 ? "left" : "right",
      wy:   slot < 2 ? -ROW_Y : ROW_Y,
      wz:   -(col * FRAME_W + 400),
    };
  });
}

function Gallery({ items, onBack }) {
  const divRef  = useRef(null);
  const camRef  = useRef({ x: 0, z: 0, yaw: 0 });
  const keysRef = useRef({});
  const rafRef  = useRef();
  const [, tick] = useState(0);

  useEffect(() => {
    const SPD = 600; // px/sec target speed
    const ACCEL = 8;  // lerp factor — higher = snappier, lower = smoother
    let velX = 0, velZ = 0;
    let last = performance.now();

    // Movement loop
    const loop = () => {
      const now = performance.now();
      const dt  = Math.min((now - last) / 1000, 0.05);
      last = now;

      const c  = camRef.current;
      const k  = keysRef.current;
      const sn = Math.sin(c.yaw), cs = Math.cos(c.yaw);
      let dx = 0, dz = 0;

      if (k["w"] || k["arrowup"])    { dx += sn; dz -= cs; }
      if (k["s"] || k["arrowdown"])  { dx -= sn; dz += cs; }
      if (k["d"] || k["arrowright"]) { dx += cs; dz += sn; }
      if (k["a"] || k["arrowleft"])  { dx -= cs; dz -= sn; }

      const len = Math.sqrt(dx*dx + dz*dz);
      if (len > 0) { dx /= len; dz /= len; }

      // Smooth velocity with lerp
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
    const onKeyUp   = (e) => { keysRef.current[e.key.toLowerCase()] = false; };
    const onMove = (e) => {
      if (document.pointerLockElement !== divRef.current) return;
      camRef.current.yaw = clamp(
        camRef.current.yaw - e.movementX * 0.003,
        -1.25, 1.25   // ±~72° — enough to see wall paintings, never look through a wall
      );
    };
    const onClick   = () => divRef.current?.requestPointerLock();

    window.addEventListener("keydown",  onKeyDown);
    window.addEventListener("keyup",    onKeyUp);
    document.addEventListener("mousemove", onMove);
    divRef.current?.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("keydown",  onKeyDown);
      window.removeEventListener("keyup",    onKeyUp);
      document.removeEventListener("mousemove", onMove);
    };
  }, [onBack]);

  const { x, z, yaw } = camRef.current;
  // World = inverse camera: undo rotation then undo translation
  const worldTransform = `rotateY(${-yaw}rad) translate3d(${-x}px, 0px, ${-z}px)`;

  return (
    <div
      ref={divRef}
      style={{
        width: "100vw", height: "100vh",
        overflow: "hidden",
        background: "#000",
        // perspective puts the eye 700px in front of Z=0
        perspective: "700px",
        perspectiveOrigin: "50% 50%",
        cursor: "crosshair",
        userSelect: "none",
      }}
    >
      {/*
        World anchor: zero-size div at screen center.
        All 3D children are positioned relative to this point.
        Camera is implicitly at this point looking toward -Z.
      */}
      <div style={{
        position: "absolute",
        left: "50%", top: "50%",
        transformStyle: "preserve-3d",
        transform: worldTransform,
      }}>

        {/* ── Floor — length covers all images ── */}
        {(() => {
          const cols = Math.ceil(items.length / 4) || 1;
          const floorLen = cols * FRAME_W * 2 + 4000;
          return (
            <div style={{
              position: "absolute",
              width: WALL_X * 2,
              height: floorLen,
              left: -WALL_X,
              top: -floorLen / 2,
              background: "linear-gradient(to bottom, #c8c4be, #d8d4cf)",
              transformStyle: "preserve-3d",
              transform: "translateY(280px) rotateX(-90deg)",
            }} />
          );
        })()}

        {/* ── Photos ── */}
        {items.map((item, i) => {
          const isLeft = item.side === "left";
          const wx     = isLeft ? -(WALL_X - 6) : (WALL_X - 6);
          const rotY   = isLeft ? "90deg" : "-90deg";
          const src    = item.thumb_url || item.path;

          return (
            <div key={i} style={{
              position: "absolute",
              // pre-center so transform-origin is at (0,0)
              left: -(IMG_W / 2 + MAT),
              top:  -(IMG_H / 2 + MAT),
              transformStyle: "preserve-3d",
              // translate to wall position then rotate to face inward
              transform: `translate3d(${wx}px, ${item.wy}px, ${item.wz}px) rotateY(${rotY})`,
            }}>
              {/* White mat */}
              <div style={{
                position: "relative",
                width:  IMG_W + MAT * 2,
                height: IMG_H + MAT * 2,
                background: "#ffffff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
              }}>
                {src && (
                  <img
                    src={src}
                    alt=""
                    style={{
                      width:  IMG_W,
                      height: IMG_H,
                      objectFit: "cover",
                      display: "block",
                    }}
                  />
                )}
                <div style={{
                  position: "absolute",
                  bottom: 3,
                  left: 0, right: 0,
                  fontSize: 9,
                  color: "#aaa",
                  fontFamily: "monospace",
                  textAlign: "center",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  padding: "0 6px",
                }}>
                  {item.image_file_id || "—"}
                </div>
              </div>
            </div>
          );
        })}

      </div>

      {/* HUD */}
      <div style={{
        position: "absolute", bottom: 16, left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,0,0,0.45)",
        color: "rgba(255,255,255,0.65)",
        padding: "8px 20px", borderRadius: 999,
        fontSize: 12, backdropFilter: "blur(8px)",
        pointerEvents: "none", whiteSpace: "nowrap",
      }}>
        Click to lock mouse · WASD to walk · Q to exit
      </div>
    </div>
  );
}

// ─── Thumbnail grid ───────────────────────────────────────────────────────────
function ThumbGrid({ results }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {results.map((row, i) => {
        const src = row.thumb_url || row.path;
        return (
          <div key={row.image_file_id || i} style={{ position: "relative", lineHeight: 0 }}>
            {src && (
              <img src={src} alt="" style={{ width: "100%", display: "block" }} />
            )}
            <div style={{
              position: "absolute", top: 4, left: 4,
              background: "rgba(0,0,0,0.55)", color: "rgba(255,255,255,0.8)",
              fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 4,
            }}>#{i + 1}</div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Root ─────────────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a SQL expert. Given a user's natural language description of images they want to find, write a PostgreSQL SELECT query against these two tables:

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
- If the user asks for something not directly in this list, map it creatively to the closest available labels. For example: "baby" → person; "breakfast" → cup + bowl + banana + orange; "office" → laptop + keyboard + mouse + chair; "kitchen scene" → sink + microwave + oven + refrigerator + bottle; "romantic dinner" → wine_glass + dining_table + cup + cake; "beach" → surfboard + umbrella + sports_ball + boat; "forest animal" → bear + bird + horse; "wallet" → handbag + tie. Always explain your mapping in point 2.
- Use yfcc_index count columns for simple presence/count filters (fast).
- Think carefully about how to use bb_table for spatial and relational queries. For example:
    - "two people kissing" → find images where person >= 2 in yfcc_index, then join bb_table to find two person bounding boxes whose regions overlap or touch (check if the distance between their centers is less than the sum of half their widths/heights).
    - "person holding a dog" → find images with person >= 1 and dog >= 1, then use bb_table to verify a person bbox and dog bbox are overlapping or adjacent.
    - "crowd" → use bb_table to find images where many person bboxes are densely packed (high count, small average bbox size).
    - "large dog" → use bb_table to filter for dog bboxes where width * height is above a threshold.
  Always reason about whether the query implies spatial proximity, relative size, or object interaction, and use bb_table accordingly.
- ORDER the results by relevance. Use confidence scores from bb_table — for example ORDER BY AVG(confidence_score) DESC or MAX(confidence_score) DESC — so the most confidently detected matches appear first. Always join bb_table for ordering even if only yfcc_index is needed for filtering. When using aggregate functions like MAX() or AVG() with a JOIN on bb_table, always GROUP BY yfcc_index.image_file_id before the ORDER BY.
- Always add LIMIT {LIMIT}.

Return exactly two points, no markdown, no backticks, no extra text:

1. The raw SQL query and nothing else — no explanation, no comments inside the SQL.
2. Exactly 2 sentences of reasoning: why you wrote the query this way, and whether {LIMIT} results is likely too few or too many for this query.`;


export default function App() {
  const [query,         setQuery]         = useState("cat dog");
  const [limit,         setLimit]         = useState(20);
  const [searchResults, setSearchResults] = useState(null);
  const [error,         setError]         = useState("");
  const [mode,          setMode]          = useState("search");
  const [galleryItems,  setGalleryItems]  = useState([]);

  const [apiKey,      setApiKey]      = useState("");
  const [sqlLoading,  setSqlLoading]  = useState(false);
  const [sqlOutput,   setSqlOutput]   = useState("");
  const [sqlError,    setSqlError]    = useState("");
  const [queryRunLoading, setQueryRunLoading] = useState(false);

  const handleEnter = () => {
    if (!searchResults?.length) return;
    setGalleryItems(makeLayout(searchResults));
    setMode("gallery");
  };

  const handleOpenAI = async () => {
    if (!query.trim()) return;
    if (!apiKey.trim()) { setSqlError("Enter an OpenAI API key first."); return; }
    setSqlLoading(true); setSqlOutput(""); setSqlError("");
    try {
      const prompt = SYSTEM_PROMPT.replaceAll("{LIMIT}", limit);
      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey.trim()}`,
        },
        body: JSON.stringify({
          model: "gpt-4o",
          messages: [
            { role: "system", content: prompt },
            { role: "user",   content: query },
          ],
          temperature: 0,
        }),
      });
      if (!res.ok) { const t = await res.text(); throw new Error(t); }
      const data = await res.json();
      setSqlOutput(data.choices?.[0]?.message?.content?.trim() || "(no output)");
    } catch (e) { setSqlError(e?.message || "OpenAI request failed"); }
    finally     { setSqlLoading(false); }
  };

  const handleRunQuery = async () => {
    if (!sqlOutput.trim()) return;
    // Extract just the SQL (point 1) — everything before the first blank line or "2."
    const sqlOnly = sqlOutput.split(/\n\s*\n|^\s*2\./m)[0].trim();
    setQueryRunLoading(true); setError("");
    try {
      const res = await fetch(`${API_BASE}/api/run_query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sql: sqlOnly }),
      });
      if (!res.ok) {
        const t = await res.json();
        throw new Error(t.error || `Server error (${res.status})`);
      }
      const data = await res.json();
      const rows = Array.isArray(data.rows) ? data.rows : [];
      if (rows.length === 0) { setError("Query ran but returned 0 results."); return; }
      setSearchResults(rows);
    } catch (e) { setError(e?.message || "Run query failed"); }
    finally     { setQueryRunLoading(false); }
  };

  if (mode === "gallery") {
    return <Gallery items={galleryItems} onBack={() => setMode("search")} />;
  }

  return (
    <div className="bg-[radial-gradient(circle_at_top,_#2a1d1a,_#120f0f_55%,_#090808)] text-white">
      <div className="px-4 pt-3 pb-6 max-w-3xl mx-auto space-y-3">
        {/* API key */}
        <input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          placeholder="sk-… API key"
          className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-white outline-none placeholder:text-white/25 focus:border-white/25 w-48"
        />
        {/* Title */}
        <h1 className="text-2xl font-bold tracking-tight text-white/90">3D Library Image Viewer</h1>
        {/* Search row */}
        <div className="flex gap-2 mt-4">
          <input value={query} onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") handleOpenAI(); }}
            placeholder="two people kissing, a dog in a park…"
            className="flex-1 rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm text-white outline-none placeholder:text-white/25 focus:border-white/25" />
          <select value={limit} onChange={e => setLimit(Number(e.target.value))}
            className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-white outline-none">
            {[20,40,60,80,100,150,200,300,500].map(n => <option key={n} value={n}>{n} images</option>)}
          </select>
          <button onClick={handleOpenAI} disabled={sqlLoading || !query.trim()}
            className="rounded-2xl border border-violet-400/40 bg-violet-400/10 px-5 py-3 text-sm font-semibold text-violet-200 transition hover:bg-violet-400/20 disabled:opacity-50 whitespace-nowrap">
            {sqlLoading ? "Thinking…" : "Generate SQL"}
          </button>
        </div>

        {/* SQL output + run button */}
        {sqlError && <p className="text-sm text-red-300">{sqlError}</p>}
        {error    && <p className="text-sm text-red-300">{error}</p>}
        {sqlOutput && (
          <>
            <pre className="rounded-2xl border border-white/8 bg-black/40 p-3 text-xs text-green-300 backdrop-blur-sm whitespace-pre-wrap break-words overflow-x-auto">
              {sqlOutput}
            </pre>
            <button onClick={handleRunQuery} disabled={queryRunLoading}
              className="w-full rounded-2xl border border-emerald-400/40 bg-emerald-400/10 px-6 py-3 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/20 disabled:opacity-50">
              {queryRunLoading ? "Running…" : "▶ Run Query on Database"}
            </button>
          </>
        )}

        {/* Results — images stacked flush, no gap */}
        {searchResults !== null && searchResults.length > 0 && (
          <>
            <div className="flex items-center justify-between pt-1">
              <p className="text-xs text-white/40">{searchResults.length} results</p>
              <button onClick={handleEnter}
                className="rounded-full border border-amber-400/40 bg-amber-400/10 px-4 py-1.5 text-xs font-semibold text-amber-200 hover:bg-amber-400/20">
                Enter Gallery →
              </button>
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              {searchResults.map((row, i) => (
                <img
                  key={row.image_file_id || i}
                  src={row.thumb_url || row.path}
                  alt=""
                  style={{ width: "400px", display: "block" }}
                />
              ))}
            </div>
          </>
        )}
        {searchResults !== null && searchResults.length === 0 && (
          <p className="text-sm text-white/35 text-center py-6">No images matched.</p>
        )}
      </div>
    </div>
  );
}