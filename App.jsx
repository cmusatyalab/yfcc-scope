import React, { useEffect, useMemo, useState } from "react";

const LABELS = ["airplane","apple","backpack","banana","baseball_bat",
  "baseball_glove","bear","bed","bench","bicycle","bird",
  "boat","book","bottle","bowl","broccoli","bus","cake",
  "car","carrot","cat","cell_phone","chair","clock","cow",
  "couch","cup","dining_table","dog","donut","elephant",
  "fire_hydrant","fork","frisbee","giraffe","hair_drier",
  "handbag","horse","hot_dog","keyboard","kite","knife",
  "laptop","microwave","motorcycle","mouse","orange","oven",
  "parking_meter","person","pizza","potted_plant","refrigerator",
  "remote","sandwich","scissors","sheep","sink","skateboard",
  "skis","snowboard","spoon","sports_ball","stop_sign",
  "suitcase","surfboard","teddy_bear","tennis_racket","tie",
  "toaster","toilet","toothbrush","traffic_light",
  "train","truck","tv","umbrella","vase","wine_glass","zebra"];

const API_BASE = "http://128.2.212.50:8080";

function binsToConfString(binSet) {
  const arr = Array.from(binSet).sort((a, b) => a - b);
  if (arr.length === 0) return "";

  const parts = [];
  let start = arr[0];
  let prev = arr[0];

  for (let i = 1; i < arr.length; i++) {
    const cur = arr[i];
    if (cur === prev + 1) {
      prev = cur;
      continue;
    }
    parts.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = cur;
    prev = cur;
  }
  parts.push(start === prev ? `${start}` : `${start}-${prev}`);
  return parts.join(",");
}

export default function App() {
  const [selectedLabels, setSelectedLabels] = useState(() => new Set());
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");

  // confidence filter string sent to /api/images as "conf"
  const [conf, setConf] = useState("");

  // histogram + selected bins (ints 0..100)
  const [hist, setHist] = useState([]);
  const [selectedBins, setSelectedBins] = useState(() => new Set());

  const selectedKey = useMemo(
    () => Array.from(selectedLabels).sort().join(","),
    [selectedLabels]
  );

  const filteredLabels = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return LABELS;
    return LABELS.filter((l) => l.includes(q));
  }, [query]);

  // ---- Fetch histogram when selected labels change ----
  useEffect(() => {
    const labs = Array.from(selectedLabels);
    if (labs.length === 0) {
      setHist([]);
      setSelectedBins(new Set());
      setConf("");
      return;
    }

    const params = new URLSearchParams();
    labs.forEach((l) => params.append("label", l));

    fetch(`${API_BASE}/api/conf_hist?${params.toString()}`)
      .then((r) => r.json())
      .then((data) => {
        setHist(Array.isArray(data.bins) ? data.bins : []);
        setSelectedBins(new Set());
        setConf("");
      })
      .catch((err) => {
        console.error("hist fetch error", err);
        setHist([]);
      });
  }, [selectedKey]);

  // ---- When selectedBins changes, update conf string ----
  useEffect(() => {
    setConf(binsToConfString(selectedBins));
  }, [selectedBins]);

  // ---- Fetch images when labels/conf changes ----
  useEffect(() => {
    let cancelled = false;

    async function run() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams();
        Array.from(selectedLabels).forEach((lab) => params.append("label", lab));
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
    return () => { cancelled = true; };
  }, [selectedKey, conf]);

  // ---- Histogram scale helpers ----
  const maxCount = useMemo(() => {
    if (!hist.length) return 1;
    return Math.max(...hist.map((b) => b.image_count || 0), 1);
  }, [hist]);

  const sortedHist = useMemo(() => {
    // ensure 0..100 order even if backend ever changes
    return [...hist].sort((a, b) => b.bin - a.bin);
  }, [hist]);

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
            Showing:{" "}
            <span style={{ fontWeight: 600, color: "#111" }}>
              {Array.from(selectedLabels).join(", ") || "none"}
            </span>
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
                  marginTop: 8,
                  width: "100%",
                  boxSizing: "border-box",
                  borderRadius: 12,
                  border: "1px solid #ddd",
                  padding: "8px 10px",
                  fontSize: 13
                }}
              />
            </div>

            <div style={{ maxHeight: "70vh", overflow: "auto", padding: 10 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {filteredLabels.map((lab) => (
                  <button
                    key={lab}
                    onClick={() => {
                      setSelectedLabels((prev) => {
                        const next = new Set(prev);
                        if (next.has(lab)) next.delete(lab);
                        else next.add(lab);
                        return next;
                      });
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
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

              {/* VERTICAL Histogram under labels */}
              <div style={{ marginTop: 16 }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>
                  Click one or multiple bars to filter and view images
                  with detections at those selected confidence levels
                </div>

                <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
                  Selected: {selectedBins.size ? binsToConfString(selectedBins) : "none"}
                </div>

                <div
                  style={{
                    marginTop: 8,
                    border: "1px solid #eee",
                    borderRadius: 12,
                    padding: 4,
                
                    // compact: fits all 0–100 bins without scrolling
                    height: 410,
                    overflowY: "hidden",
                    background: "#fafafa",
                
                    display: "flex",
                    flexDirection: "column",
                    gap: 0
                  }}
                >
                  {sortedHist.map(({ bin, image_count }) => {
                    const selected = selectedBins.has(bin);
                
                    // log scale so low bins are still visible
                    const w = Math.max(
                      1,
                      (Math.log10((image_count || 0) + 1) / Math.log10(maxCount + 1)) * 100
                    );
                
                    return (
                      <div
                        key={bin}
                        onClick={() => {
                          setSelectedBins((prev) => {
                            const next = new Set(prev);
                            if (next.has(bin)) next.delete(bin);
                            else next.add(bin);
                            return next;
                          });
                        }}
                        title={`${bin}% — ${image_count} images`}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          cursor: "pointer",
                
                          // ultra compact rows (101 bins fit in ~200px)
                          height: 4,
                          padding: 0,
                          margin: 0,
                          borderRadius: 0,
                          background: selected ? "rgba(255,77,79,0.2)" : "transparent"
                        }}
                      >
                        {/* Show bin label only every 10 to save space */}
                        {bin % 10 === 0 ? (
                          <div
                            style={{
                              width: 20,
                              fontSize: 9,
                              color: "#777",
                              textAlign: "right",
                              marginRight: 4
                            }}
                          >
                            {bin}
                          </div>
                        ) : (
                          <div style={{ width: 20, marginRight: 4 }} />
                        )}
                
                        <div
                          style={{
                            flex: 1,
                            height: 4,
                            background: "rgba(0,0,0,0.06)",
                            borderRadius: 0
                          }}
                        >
                          <div
                            style={{
                              width: `${w}%`,
                              height: "100%",
                              borderRadius: 0,
                              background: selected ? "#ff4d4f" : "#999"
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                
                  {sortedHist.length === 0 && (
                    <div style={{ padding: 10, fontSize: 13, color: "#666" }}>
                      Select at least one label to load the histogram.
                    </div>
                  )}
                </div>

                  

                {selectedBins.size > 0 && (
                  <button
                    onClick={() => setSelectedBins(new Set())}
                    style={{
                      marginTop: 8,
                      width: "100%",
                      borderRadius: 12,
                      border: "1px solid #ddd",
                      padding: "8px 10px",
                      fontSize: 13,
                      cursor: "pointer",
                      background: "white"
                    }}
                  >
                    Clear confidence selection
                  </button>
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
