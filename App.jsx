import React, { useEffect, useMemo, useState } from "react";

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
  "clock","vase","scissors","teddy_bear","hair_drier","toothbrush"
];

// since you're running React on :5173 and API on :8080
const API_BASE = "http://128.2.212.50:8080";

function clsx(...xs) {
  return xs.filter(Boolean).join(" ");
}

export default function App() {
  const [selectedLabels, setSelectedLabels] = useState(() => new Set());
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");

  const selectedKey = useMemo(
    () => Array.from(selectedLabels).sort().join(","),
    [selectedLabels]
  );
  // optional confidence filter input: "40-45,50,90-100"
  const [conf, setConf] = useState("");
  

  const filteredLabels = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return LABELS;
    return LABELS.filter((l) => l.includes(q));
  }, [query]);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams();
        Array.from(selectedLabels).forEach(lab => params.append("label", lab));;
        params.set("limit", "50");
        if (conf.trim()) params.set("conf", conf.trim());

        const url = `${API_BASE}/api/images?${params.toString()}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setImages(Array.isArray(data.images) ? data.images : []);
      } catch (e) {
        if (cancelled) return;
        setError(e?.message || "Failed to load images");
        setImages([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [selectedKey, conf]);

  return (
    <div style={{ minHeight: "100vh", background: "#f5f5f7", color: "#111" }}>
      <div style={{ width: "100%", maxWidth: "100%", padding: 16, boxSizing: "border-box" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>Dashboard</div>
            <div style={{ fontSize: 13, color: "#666" }}>
              Select labels to show the first 50 images with at least one bbox from any selected label.
            </div>
          </div>
          <div style={{ fontSize: 13, color: "#666" }}>
            Showing: <span style={{ fontWeight: 600, color: "#111" }}>  {Array.from(selectedLabels).join(", ") || "none"}</span>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 12 }}>
          {/* Sidebar */}
          <aside style={{ background: "white", borderRadius: 16, border: "1px solid rgba(0,0,0,0.08)" }}>
            <div style={{ padding: 12, borderBottom: "1px solid rgba(0,0,0,0.08)" }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>Labels</div>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search labels…"
                style={{
                  marginTop: 8, width: "100%", boxSizing: "border-box", borderRadius: 12, border: "1px solid #ddd",
                  padding: "8px 10px", fontSize: 13
                }}
              />
            </div>

            <div style={{ maxHeight: "70vh", overflow: "auto", padding: 10 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {filteredLabels.map((lab) => (
                  <button
                      key={lab}
                      onClick={() => {
                          setSelectedLabels(prev => {
                            const next = new Set(prev);
                            if (next.has(lab)) next.delete(lab);
                            else next.add(lab);
                            return next;
                          });
                        }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8, // spacing between dot and label
                        borderRadius: 12,
                        border: "1px solid",
                        borderColor: selectedLabels.has(lab) ? "#111" : "#e5e5e5",
                        background: selectedLabels.has(lab) ? "#111" : "white",
                        color: selectedLabels.has(lab) ? "white" : "#111",
                        padding: "8px 10px",
                        fontSize: 13,
                        cursor: "pointer"
                      }}
                    >
                      <span
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: 999,
                          background: selectedLabels.has(lab) ? "white" : "#ccc",
                          flexShrink: 0
                        }}
                      />
                      <span style={{ fontWeight: 600 }}>{lab}</span>
                    </button>

                ))}
                {filteredLabels.length === 0 && (
                  <div style={{ padding: 12, border: "1px dashed #ccc", borderRadius: 12, color: "#666", fontSize: 13 }}>
                    No labels match “{query}”.
                  </div>
                )}
              </div>
            </div>
          </aside>

          {/* Main */}
          <main style={{ background: "white", borderRadius: 16, border: "1px solid rgba(0,0,0,0.08)", padding: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700 }}>Images (up to 50)</div>
              <div style={{ fontSize: 12, color: "#666" }}>{loading ? "Loading…" : `${images.length} loaded`}</div>
            </div>

            {/* Confidence filter */}
            {/* <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10 }}>
              <div style={{ fontSize: 12, color: "#666", fontWeight: 600 }}>Confidence filter (0–100):</div>
              <input
                value={conf}
                onChange={(e) => setConf(e.target.value)}
                placeholder='e.g. 40-45,50,90-100'
                style={{
                  flex: 1, borderRadius: 12, border: "1px solid #ddd",
                  padding: "8px 10px", fontSize: 13
                }}
              />
              <button
                onClick={() => setConf("")}
                style={{ borderRadius: 12, border: "1px solid #ddd", padding: "8px 10px", fontSize: 13, cursor: "pointer" }}
              >
                Clear
              </button>
            </div> */}

            {error && (
              <div style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 12, border: "1px solid #f2b8b8", background: "#fff1f1", color: "#a40000", fontSize: 13 }}>
                {error}
              </div>
            )}

            {images.length === 0 && !loading ? (
              <div style={{ padding: 20, borderRadius: 12, border: "1px dashed #ccc", color: "#666", fontSize: 13 }}>
                No images found for “{Array.from(selectedLabels).join(", ") || "none"}”.
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 8 }}>
                {images.map((img) => (
                  <a
                    key={img.image_file_id}
                    href={`${API_BASE}/?image_file_id=${encodeURIComponent(img.image_file_id)}&select_all=0&${Array.from(selectedLabels).map(l => `label=${encodeURIComponent(l)}`).join("&")}`}
                    target="_blank"
                    rel="noreferrer"
                    title={img.image_file_id}
                    style={{
                      borderRadius: 12,
                      overflow: "hidden",
                      border: "1px solid #e5e5e5",
                      display: "block",
                      textDecoration: "none"
                    }}
                  >
                    <img
                      src={img.path}
                      alt={img.image_file_id}
                      loading="lazy"
                      style={{ width: "100%", aspectRatio: "4/3", objectFit: "cover", display: "block" }}
                    />
                  </a>
                ))}
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
