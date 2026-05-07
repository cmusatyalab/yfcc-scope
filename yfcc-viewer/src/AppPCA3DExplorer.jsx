// Visualize all 8,000 fetched images simultaneously as a 3D point cloud.

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

/**
 * 3D PCA Explorer (LAION-style)
 * - Fetch high-dim vectors (bbox-count vectors) from your backend
 * - Run PCA -> 3D coords
 * - Color by semantic category (plants/fruit green, animals blue, etc.)
 *
 * Assumed backend payload shape (change to match your API):
 * GET `${API_BASE}/api/vector_rows?limit=5000`
 * {
 *   rows: [
 *     {
 *       id: "optional",
 *       image_file_id: "abc",
 *       thumb_url: "http://.../thumb.jpg", // optional
 *       // counts vector (COCO-style)
 *       total_bboxes: 6,
 *       person: 4,
 *       car: 1,
 *       banana: 1,
 *       ... all other labels ...
 *     },
 *     ...
 *   ]
 * }
 */

const API_BASE = "http://128.2.212.50:8080";

/** Feature order: total_bboxes + COCO labels (match your column order) */
const FEATURES = [
  "total_bboxes",
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic_light",
  "fire_hydrant",
  "stop_sign",
  "parking_meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports_ball",
  "kite",
  "baseball_bat",
  "baseball_glove",
  "skateboard",
  "surfboard",
  "tennis_racket",
  "bottle",
  "wine_glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot_dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted_plant",
  "bed",
  "dining_table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell_phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy_bear",
  "hair_drier",
  "toothbrush",
];

/** Categories (add/adjust freely) */
const CATEGORY_RULES = [
  {
    key: "plants_fruit",
    name: "Plants & Fruit",
    color: "#2ecc71",
    labels: ["apple", "banana", "orange", "broccoli", "carrot", "potted_plant"],
  },
  {
    key: "animals",
    name: "Animals",
    color: "#3498db",
    labels: ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
  },
  {
    key: "vehicles",
    name: "Vehicles",
    color: "#f39c12",
    labels: ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
  },
  {
    key: "furniture_home",
    name: "Furniture & Home",
    color: "#9b59b6",
    labels: ["chair", "couch", "bed", "dining_table", "toilet", "sink", "refrigerator", "vase", "clock"],
  },
  {
    key: "electronics",
    name: "Electronics",
    color: "#e84393",
    labels: ["tv", "laptop", "mouse", "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster"],
  },
  {
    key: "kitchen_food",
    name: "Kitchen & Food",
    color: "#e74c3c",
    labels: ["bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "sandwich", "hot_dog", "pizza", "donut", "cake"],
  },
  {
    key: "people",
    name: "People",
    color: "#FFFFFF",
    labels: ["person"],
  },
];

const DEFAULT_CATEGORY = { key: "other", name: "Other", color: "#95a5a6" };

/** Utilities */
function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}
function log1p(x) {
  return Math.log(1 + (x || 0));
}

/**
 * Infer a single category per row.
 * Priority order = CATEGORY_RULES order above.
 */
function inferCategory(row) {
  const n = (k) => Number(row?.[k] ?? 0); // handles undefined + string "0"/"1"

  for (const rule of CATEGORY_RULES) {
    for (const lab of rule.labels) {
      if (n(lab) > 0) return rule;
    }
  }
  return DEFAULT_CATEGORY;
}

/** Build numeric vector from row in FEATURES order */
function rowToVector(row) {
  return FEATURES.map((k) => log1p(row[k] || 0));
}

/**
 * PCA (top-k) via power iteration on covariance matrix:
 * - Center X
 * - Compute covariance C = (X^T X)/(n-1)
 * - Extract k eigenvectors with deflation
 * This is simple + dependency-free. Works well for moderate D (≈ 81) and n up to a few 10k.
 */
function pcaTopK(X, k = 3, iters = 50) {
  const n = X.length;
  const d = X[0]?.length || 0;
  if (!n || !d) return { mean: [], components: [], projected: [] };

  // mean
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  // centered
  const Xc = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(d);
    for (let j = 0; j < d; j++) row[j] = X[i][j] - mean[j];
    Xc[i] = row;
  }

  // covariance C = Xc^T Xc / (n-1)
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const xi = Xc[i];
    for (let a = 0; a < d; a++) {
      const va = xi[a];
      if (va === 0) continue;
      for (let b = 0; b < d; b++) C[a][b] += va * xi[b];
    }
  }
  const denom = Math.max(1, n - 1);
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] /= denom;

  function matVec(M, v) {
    const out = new Array(M.length).fill(0);
    for (let i = 0; i < M.length; i++) {
      let s = 0;
      const Mi = M[i];
      for (let j = 0; j < v.length; j++) s += Mi[j] * v[j];
      out[i] = s;
    }
    return out;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function norm(v) {
    return Math.sqrt(dot(v, v)) || 1;
  }

  function normalize(v) {
    const nrm = norm(v);
    for (let i = 0; i < v.length; i++) v[i] /= nrm;
    return v;
  }

  // Copy covariance for deflation
  let M = C.map((r) => r.slice());
  const components = [];

  for (let comp = 0; comp < k; comp++) {
    // init random vector
    let v = new Array(d).fill(0).map(() => (Math.random() - 0.5));
    normalize(v);

    for (let t = 0; t < iters; t++) {
      v = matVec(M, v);
      normalize(v);
    }

    // eigenvalue approx
    const Mv = matVec(M, v);
    const lambda = dot(v, Mv);

    components.push(v.slice());

    // deflation: M = M - lambda * v v^T
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        M[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  // project: Y = Xc * components^T  -> n x k
  const projected = new Array(n);
  for (let i = 0; i < n; i++) {
    const yi = new Array(k).fill(0);
    for (let c = 0; c < k; c++) {
      const v = components[c];
      let s = 0;
      for (let j = 0; j < d; j++) s += Xc[i][j] * v[j];
      yi[c] = s;
    }
    projected[i] = yi;
  }

  return { mean, components, projected };
}

/** Scale projected coords to a nice cube */
function normalizeCoords3D(P) {
  if (!P.length) return P;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const [x, y, z] of P) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;

  const sx = (maxX - minX) || 1;
  const sy = (maxY - minY) || 1;
  const sz = (maxZ - minZ) || 1;

  const s = Math.max(sx, sy, sz);

  return P.map(([x, y, z]) => [((x - cx) / s) * 60, ((y - cy) / s) * 60, ((z - cz) / s) * 60]);
}

/** 3D Points (instanced) */
function PointsCloud({ points, pointSize, opacity, onPick }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();

    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      positions[i * 3 + 0] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      const c = new THREE.Color(p.color);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return g;
  }, [points]);

  const mat = useMemo(() => {
    return new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity,
      depthWrite: false,
    });
  }, [pointSize, opacity]);

  useEffect(() => {
    mat.size = pointSize;
    mat.opacity = opacity;
  }, [mat, pointSize, opacity]);

  return (
    <points
      geometry={geom}
      material={mat}
      onPointerDown={(e) => {
        e.stopPropagation();
        if (e.index == null) return; // <-- THIS is the point index
        onPick?.(e.index);
      }}
    />
  );
}


export default function App() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // UI filters (LAION-ish)
  const [categoryEnabled, setCategoryEnabled] = useState(() => {
    const obj = {};
    for (const c of CATEGORY_RULES) obj[c.key] = true;
    obj[DEFAULT_CATEGORY.key] = true;
    return obj;
  });

  const [showLabels, setShowLabels] = useState(false);
  const [density, setDensity] = useState(0.35); // 0..1 (subsample)
  const [pointSize, setPointSize] = useState(0.9);

  const [picked, setPicked] = useState(null); // picked point data

  // Fetch rows
  useEffect(() => {
    let cancelled = false;

    async function run() {
      setLoading(true);
      setErr("");
      try {
        const url = `${API_BASE}/api/vector_rows?limit=8000`; // <-- change to your endpoint
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        const data = await res.json();
        const r = Array.isArray(data.rows) ? data.rows : [];
        console.log("vector_rows fetched:", r.length, r[0]); // <-- add this
        if (!cancelled) setRows(r);
      } catch (e) {
        if (!cancelled) setErr(e?.message || "Failed to load vectors");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, []);

  // Compute PCA + points
  const points = useMemo(() => {
    if (!rows.length) return [];

    // Subsample for density
    const kept = [];
    for (let i = 0; i < rows.length; i++) {
      if (Math.random() <= density) kept.push(rows[i]);
    }
    // guarantee at least something
    if (!kept.length) kept.push(rows[0]);

    const X = kept.map(rowToVector);
    const { projected } = pcaTopK(X, 3, 60);
    const coords = normalizeCoords3D(projected);

    const out = kept.map((r, i) => {
      const cat = inferCategory(r);
      return {
        idx: i,
        id: r.id ?? r.image_file_id ?? `row-${i}`,
        image_file_id: r.image_file_id,
        thumb_url: r.thumb_url || r.path || null,
        raw: r,
        category: cat.key,
        categoryName: cat.name,
        color: cat.color,
        x: coords[i][0],
        y: coords[i][1],
        z: coords[i][2],
      };
    });

    // Apply category filters
    return out.filter((p) => categoryEnabled[p.category] ?? true);
  }, [rows, density, categoryEnabled]);

  const categoryCounts = useMemo(() => {
    const counts = {};
    for (const c of CATEGORY_RULES) counts[c.key] = 0;
    counts[DEFAULT_CATEGORY.key] = 0;

    for (const r of rows) {
      const cat = inferCategory(r);
      counts[cat.key] = (counts[cat.key] || 0) + 1;
    }
    return counts;
  }, [rows]);

  return (
    <div style={{ height: "100vh", width: "100vw", background: "#0b0e14", color: "#eaeef7" }}>
      {/* Top bar */}
      <div
        style={{
          height: 48,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 14px",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          background: "linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <div style={{ fontWeight: 800, letterSpacing: 0.2, color: "#ffffff" }}>
            PCA 3D Dataset Explorer
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "rgba(255,255,255,0.75)" }}>
          {loading ? "Loading vectors…" : `${rows.length.toLocaleString()} rows`}
          {err && <span style={{ color: "#ff7675" }}>• {err}</span>}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr 280px", height: "calc(100vh - 48px)" }}>
        {/* Left sidebar */}
        <aside
          style={{
            borderRight: "1px solid rgba(255,255,255,0.08)",
            background: "rgba(255,255,255,0.02)",
            padding: 12,
            overflow: "auto",
          }}
        >
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", marginBottom: 10 }}>Dataset Clusters</div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[...CATEGORY_RULES, DEFAULT_CATEGORY].map((c) => {
              const enabled = categoryEnabled[c.key] ?? true;
              const count = categoryCounts[c.key] || 0;
              return (
                <button
                  key={c.key}
                  onClick={() =>
                    setCategoryEnabled((prev) => ({
                      ...prev,
                      [c.key]: !enabled,
                    }))
                  }
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: 10,
                    padding: "10px 10px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.10)",
                    background: enabled ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.02)",
                    color: "rgba(255,255,255,0.92)",
                    cursor: "pointer",
                    textAlign: "left",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: 999,
                        background: c.color,
                        boxShadow: "0 0 0 3px rgba(255,255,255,0.06)",
                        opacity: enabled ? 1 : 0.35,
                      }}
                    />
                    <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.1 }}>
                      <span style={{ fontWeight: 700, fontSize: 13 }}>{c.name}</span>
                      <span style={{ fontSize: 11, color: "rgba(255,255,255,0.6)" }}>{count.toLocaleString()} rows</span>
                    </div>
                  </div>
                  <span style={{ fontSize: 11, color: enabled ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.35)" }}>
                    {enabled ? "ON" : "OFF"}
                  </span>
                </button>
              );
            })}
          </div>

          <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", marginBottom: 8 }}>Controls</div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
                  <span style={{ color: "rgba(255,255,255,0.75)" }}>Density</span>
                  <span style={{ color: "rgba(255,255,255,0.75)" }}>{Math.round(density * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={density}
                  onChange={(e) => setDensity(parseFloat(e.target.value))}
                  style={{ width: "100%" }}
                />
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginTop: 4 }}>
                  Subsamples points for performance & readability.
                </div>
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
                  <span style={{ color: "rgba(255,255,255,0.75)" }}>Point size</span>
                  <span style={{ color: "rgba(255,255,255,0.75)" }}>{pointSize.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={0.4}
                  max={2.0}
                  step={0.1}
                  value={pointSize}
                  onChange={(e) => setPointSize(parseFloat(e.target.value))}
                  style={{ width: "100%" }}
                />
              </div>

              <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(255,255,255,0.78)" }}>
                <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />
                Show hover labels
              </label>
            </div>
          </div>
        </aside>

        {/* Center: 3D plot */}
        <main style={{ position: "relative" }}>
          <Canvas camera={{ position: [0, 0, 120], fov: 55 }}>
            <ambientLight intensity={0.6} />
            <pointLight position={[50, 50, 50]} intensity={0.8} />
            <OrbitControls enableDamping dampingFactor={0.08} />

            {/* faint grid box vibe */}
            <gridHelper args={[180, 18, "#1f2a44", "#141b2d"]} />

            <PointsCloud
              points={points}
              pointSize={pointSize}
              opacity={0.85}
              onPick={(idx) => {
                const p = points[idx];
                if (!p) return;
                setPicked(p);
              }}
            />

            {/* Hover labels (cheap version: show for picked only; hover can be added with raycasting if you want) */}
            {picked && showLabels && (
              <Html position={[picked.x, picked.y, picked.z]} center>
                <div
                  style={{
                    background: "rgba(0,0,0,0.75)",
                    border: "1px solid rgba(255,255,255,0.18)",
                    color: "white",
                    padding: "6px 8px",
                    borderRadius: 10,
                    fontSize: 12,
                    maxWidth: 220,
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: 2 }}>{picked.categoryName}</div>
                  <div style={{ opacity: 0.85 }}>{picked.image_file_id || picked.id}</div>
                </div>
              </Html>
            )}
          </Canvas>

          {/* mini hint overlay */}
          <div
            style={{
              position: "absolute",
              left: 12,
              bottom: 12,
              padding: "8px 10px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(255,255,255,0.03)",
              fontSize: 12,
              color: "rgba(255,255,255,0.75)",
            }}
          >
            Drag to rotate • Scroll to zoom • Click a point to inspect
          </div>
        </main>

        {/* Right panel: point inspector */}
        <aside
          style={{
            borderLeft: "1px solid rgba(255,255,255,0.08)",
            background: "rgba(255,255,255,0.02)",
            padding: 12,
            overflow: "auto",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)" }}>Inspector</div>
            <button
              onClick={() => setPicked(null)}
              style={{
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.03)",
                color: "rgba(255,255,255,0.8)",
                borderRadius: 10,
                padding: "6px 8px",
                fontSize: 12,
                cursor: "pointer",
              }}
            >
              Clear
            </button>
          </div>

          {!picked ? (
            <div
              style={{
                marginTop: 10,
                padding: 12,
                borderRadius: 14,
                border: "1px dashed rgba(255,255,255,0.18)",
                color: "rgba(255,255,255,0.65)",
                fontSize: 13,
              }}
            >
              Click a point to see details.
            </div>
          ) : (
            <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 10 }}>
              <div
                style={{
                  border: "1px solid rgba(255,255,255,0.10)",
                  background: "rgba(255,255,255,0.03)",
                  borderRadius: 16,
                  padding: 12,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 999, background: picked.color }} />
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14 }}>{picked.categoryName}</div>
                    <div style={{ fontSize: 12, color: "rgba(255,255,255,0.65)" }}>{picked.image_file_id || picked.id}</div>
                  </div>
                </div>

                {picked.thumb_url ? (
                  <img
                    src={picked.thumb_url}
                    alt=""
                    style={{
                      width: "100%",
                      marginTop: 10,
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.10)",
                      display: "block",
                    }}
                  />
                ) : (
                  <div style={{ marginTop: 10, fontSize: 12, color: "rgba(255,255,255,0.55)" }}>
                    No thumbnail in payload (add <code style={{ color: "rgba(255,255,255,0.8)" }}>thumb_url</code> or{" "}
                    <code style={{ color: "rgba(255,255,255,0.8)" }}>path</code>).
                  </div>
                )}

                <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                  <a
                    href={
                      picked.image_file_id
                        ? `${API_BASE}/?image_file_id=${encodeURIComponent(picked.image_file_id)}&select_all=0`
                        : "#"
                    }
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      flex: 1,
                      textAlign: "center",
                      textDecoration: "none",
                      padding: "8px 10px",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.12)",
                      background: "rgba(255,255,255,0.04)",
                      color: "rgba(255,255,255,0.85)",
                      fontSize: 12,
                      pointerEvents: picked.image_file_id ? "auto" : "none",
                      opacity: picked.image_file_id ? 1 : 0.5,
                    }}
                  >
                    Open image
                  </a>

                  <button
                    onClick={() => {
                      // “Random paper” vibe: pick a random currently-visible point
                      if (!points.length) return;
                      setPicked(points[Math.floor(Math.random() * points.length)]);
                    }}
                    style={{
                      flex: 1,
                      textAlign: "center",
                      padding: "8px 10px",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.12)",
                      background: "rgba(255,255,255,0.04)",
                      color: "rgba(255,255,255,0.85)",
                      fontSize: 12,
                      cursor: "pointer",
                    }}
                  >
                    Random point
                  </button>
                </div>
              </div>

              {/* Show top nonzero bbox counts */}
              <div
                style={{
                  border: "1px solid rgba(255,255,255,0.10)",
                  background: "rgba(255,255,255,0.03)",
                  borderRadius: 16,
                  padding: 12,
                }}
              >
                <div style={{ fontWeight: 800, marginBottom: 8, fontSize: 13 }}>BBox counts</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {FEATURES.filter((k) => k !== "total_bboxes")
                    .map((k) => ({ k, v: picked.raw?.[k] || 0 }))
                    .filter((x) => x.v > 0)
                    .sort((a, b) => b.v - a.v)
                    .slice(0, 12)
                    .map(({ k, v }) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                        <span style={{ color: "rgba(255,255,255,0.78)" }}>{k}</span>
                        <span style={{ color: "rgba(255,255,255,0.60)" }}>{v}</span>
                      </div>
                    ))}
                  {picked.raw?.total_bboxes != null && (
                    <div style={{ marginTop: 8, fontSize: 12, color: "rgba(255,255,255,0.65)" }}>
                      total_bboxes: <span style={{ color: "rgba(255,255,255,0.9)", fontWeight: 700 }}>{picked.raw.total_bboxes}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Legend */}
          <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            <div style={{ fontSize: 12, color: "rgba(255,255,255,0.7)", marginBottom: 8 }}>Legend</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {CATEGORY_RULES.map((c) => (
                <div key={c.key} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(255,255,255,0.75)" }}>
                  <span style={{ width: 10, height: 10, borderRadius: 999, background: c.color }} />
                  <span>{c.name}</span>
                </div>
              ))}
              <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "rgba(255,255,255,0.75)" }}>
                <span style={{ width: 10, height: 10, borderRadius: 999, background: DEFAULT_CATEGORY.color }} />
                <span>{DEFAULT_CATEGORY.name}</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
