import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Color, Vector3 } from "three";
import "./AppClusterExplorer.css";

// Read API base from env
const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const API_PREFIX = `${API_BASE}/api`;

// Fetch the 3D PCA centroids from the API
async function fetchPca3dCentroids() {
  const response = await fetch(`${API_PREFIX}/centroids_pca3d`);
  if (!response.ok) {
    throw new Error(
      `Failed to load centroids: ${response.status} ${response.statusText}`,
    );
  }
  return await response.json();
}

// Fetch the cluster sizes from the API
async function fetchClusterSizes() {
  const response = await fetch(`${API_PREFIX}/cluster_sizes`);
  if (!response.ok) {
    throw new Error(
      `Failed to load cluster sizes: ${response.status} ${response.statusText}`,
    );
  }
  return await response.json();
}

async function fetchClusterImageIdx(clusterIndex) {
  const response = await fetch(
    `${API_PREFIX}/cluster_image_indexes?cluster=${clusterIndex}`,
  );
  if (!response.ok) {
    throw new Error(
      `Failed to load cluster image indexes: ${response.status} ${response.statusText}`,
    );
  }
  return await response.json();
}

function TopBar({ Message }) {
  return (
    <div className="cluster-explorer-topbar">
      <div className="cluster-explorer-topbar-title">{Message}</div>
    </div>
  );
}

const BUCKET_COLOR_HEX = [
  "#d55e00",
  "#e69f00",
  "#009e73",
  "#0072b2",
  "#9f61d4",
];

const BUCKET_COLORS = BUCKET_COLOR_HEX.map((hex) => {
  const color = new Color(hex);
  return [color.r, color.g, color.b];
});

function sizeToBucket(size) {
  const safeSize = Math.max(1, Number(size));
  return Math.min(4, Math.max(0, Math.round(Math.log10(safeSize))));
}

function sizeToRgb(size) {
  return BUCKET_COLORS[sizeToBucket(size)];
}

function buildPointData(centroids, clusterSizes, bucketFilter = null) {
  const positions = [];
  const colors = [];

  for (let i = 0; i < centroids.length; i++) {
    const size = Number(clusterSizes[i]);
    const bucket = sizeToBucket(size);

    if (bucketFilter !== null && bucket !== bucketFilter) {
      continue;
    }

    positions.push(centroids[i][0], centroids[i][1], centroids[i][2]);

    const [red, green, blue] = sizeToRgb(size);
    colors.push(red, green, blue);
  }

  return {
    positions: new Float32Array(positions),
    colors: new Float32Array(colors),
  };
}

function PointsLayer({
  positions,
  colors,
  size,
  color,
  opacity,
  depthWrite,
  renderOrder,
  onPointClick,
}) {
  return (
    <points renderOrder={renderOrder} onPointerDown={onPointClick}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        {colors ? (
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        ) : null}
      </bufferGeometry>
      <pointsMaterial
        size={size}
        color={color}
        vertexColors={colors ? true : undefined}
        sizeAttenuation
        transparent={opacity < 1}
        opacity={opacity}
        depthWrite={depthWrite}
      />
    </points>
  );
}

function PointCloudWithHighlight({
  centroids,
  clusterSizes,
  selectedBucket,
  setSelectedCentroid,
}) {
  const basePoints = useMemo(
    () => buildPointData(centroids, clusterSizes),
    [centroids, clusterSizes],
  );
  const highlightedPoints = useMemo(
    () =>
      selectedBucket === null
        ? null
        : buildPointData(centroids, clusterSizes, selectedBucket),
    [centroids, clusterSizes, selectedBucket],
  );

  return (
    <>
      <PointsLayer
        positions={basePoints.positions}
        colors={basePoints.colors}
        size={0.05}
        opacity={selectedBucket === null ? 1 : 0.5}
        depthWrite={true}
        renderOrder={1}
        onPointClick={(event) => {
          event.stopPropagation();
          setSelectedCentroid(event.index);
        }}
      />
      {highlightedPoints ? (
        <>
          <PointsLayer
            positions={highlightedPoints.positions}
            size={0.15}
            color="#ffffff"
            opacity={0.7}
            depthWrite={true}
            renderOrder={0}
          />
        </>
      ) : null}
    </>
  );
}

function KeyboardCameraControls({ selectedCentroid, centroids }) {
  const { camera } = useThree();
  const controlsRef = useRef(null);
  const pressedKeysRef = useRef(new Set());
  const forwardRef = useRef(new Vector3());
  const rightRef = useRef(new Vector3());
  const movementRef = useRef(new Vector3());

  useEffect(() => {
    const handleKeyDown = (event) => {
      pressedKeysRef.current.add(event.key.toLowerCase());
    };

    const handleKeyUp = (event) => {
      pressedKeysRef.current.delete(event.key.toLowerCase());
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useFrame((_, delta) => {
    const pressedKeys = pressedKeysRef.current;
    const moveSpeed = 5 * delta;
    const offsetRight =
      (pressedKeys.has("d") || pressedKeys.has("arrowright") ? 1 : 0) -
      (pressedKeys.has("a") || pressedKeys.has("arrowleft") ? 1 : 0);
    const offsetForward =
      (pressedKeys.has("w") || pressedKeys.has("arrowup") ? 1 : 0) -
      (pressedKeys.has("s") || pressedKeys.has("arrowdown") ? 1 : 0);
    const focusTarget = pressedKeys.has("f");
    const shouldMove = offsetRight !== 0 || offsetForward !== 0;

    if (!shouldMove && !focusTarget) {
      return;
    }

    if (focusTarget) {
      if (controlsRef.current && selectedCentroid !== null && centroids) {
        controlsRef.current.target.set(
          centroids[selectedCentroid][0],
          centroids[selectedCentroid][1],
          centroids[selectedCentroid][2],
        );
      }
      controlsRef.current.update();
      return;
    }

    const forward = forwardRef.current;
    const right = rightRef.current;
    const movement = movementRef.current;

    camera.getWorldDirection(forward).normalize();
    right.crossVectors(forward, camera.up).normalize();

    movement
      .copy(right)
      .multiplyScalar(offsetRight)
      .addScaledVector(forward, offsetForward)
      .multiplyScalar(moveSpeed);

    camera.position.add(movement);

    if (controlsRef.current) {
      controlsRef.current.target.add(movement);
      controlsRef.current.update();
    }
  });

  return <OrbitControls ref={controlsRef} />;
}

function buildClusterImageUrl(imageIdx) {
  return `${API_PREFIX}/image_wds?image_idx=${imageIdx}`;
}

function SideBarContent({ selectedCentroid, clusterSizes }) {
  const [imageIdxs, setImageIdxs] = useState([]);

  useEffect(() => {
    if (selectedCentroid === null) {
      setImageIdxs([]);
      return;
    }

    async function loadClusterImageIdx(clusterIndex) {
      try {
        const imageIdx = await fetchClusterImageIdx(clusterIndex);
        setImageIdxs(imageIdx);
      } catch (error) {
        console.error("Error fetching cluster image indexes:", error);
      }
    }

    setImageIdxs([]);
    loadClusterImageIdx(selectedCentroid);
  }, [selectedCentroid]);

  if (selectedCentroid === null) {
    return <p>No centroid selected</p>;
  }

  const visibleImageIdxs = imageIdxs.slice(0, 10);

  return (
    <>
      <p className="cluster-explorer-sidebar-summary">
        Cluster Index: {selectedCentroid}
      </p>
      <p className="cluster-explorer-sidebar-summary">
        Cluster Size: {clusterSizes[selectedCentroid]}
      </p>
      <p className="cluster-explorer-sidebar-summary">
        Showing {visibleImageIdxs.length} images
      </p>
      <div className="cluster-explorer-results-list">
        {visibleImageIdxs.map((imageIdx) => (
          <div key={imageIdx} className="cluster-explorer-result-item">
            <div className="cluster-explorer-result-card">
              <div className="cluster-explorer-img-container">
                <img
                  src={buildClusterImageUrl(imageIdx)}
                  className="cluster-explorer-result-img"
                  loading="lazy"
                />
              </div>

              <div className="cluster-explorer-result-meta">
                <div>Image Index: {imageIdx}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

function SideBar({ selectedCentroid, clusterSizes }) {
  return (
    <div className="cluster-explorer-sidebar">
      <h3 className="cluster-explorer-sidebar-title">Selected Centroid</h3>
      <div className="cluster-explorer-sidebar-content">
        <SideBarContent
          selectedCentroid={selectedCentroid}
          clusterSizes={clusterSizes}
        />
      </div>
    </div>
  );
}

function Legend({ selectedBucket, setSelectedBucket }) {
  return (
    <div className="cluster-explorer-legend" aria-label="Cluster size legend">
      <div className="cluster-explorer-legend-title">
        Rounded log10(cluster size)
      </div>
      <div className="cluster-explorer-legend-buttons">
        {BUCKET_COLOR_HEX.map((color, bucket) => (
          <button
            key={color}
            type="button"
            className={`cluster-explorer-legend-button${
              selectedBucket === bucket ? " is-active" : ""
            }`}
            style={{ backgroundColor: color }}
            onClick={() => {
              setSelectedBucket((current) =>
                current === bucket ? null : bucket,
              );
            }}
            aria-pressed={selectedBucket === bucket}
            aria-label={`Highlight bucket ${bucket}`}
          >
            {bucket}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const centroidsRef = useRef([]);
  const clusterSizesRef = useRef([]);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [selectedBucket, setSelectedBucket] = useState(null);
  const [selectedCentroid, setSelectedCentroid] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const centroids = await fetchPca3dCentroids();
        const clusterSizes = await fetchClusterSizes();
        centroidsRef.current = centroids;
        clusterSizesRef.current = clusterSizes;
        console.log(
          "cluster explorer centroids:",
          centroids.length,
          centroids.slice(0, 10),
        );
        console.log(
          "cluster explorer cluster sizes:",
          clusterSizes.length,
          clusterSizes.slice(0, 10),
        );
        setDataLoaded(true);
      } catch (error) {
        setLoadError(error instanceof Error ? error.message : String(error));
      }
    }

    loadData(centroidsRef, clusterSizesRef, setDataLoaded, setLoadError);
  }, []);

  if (loadError) {
    return <TopBar Message={`Failed to load cluster explorer: ${loadError}`} />;
  }
  if (!dataLoaded) {
    return <TopBar Message="Loading 3D PCA data..." />;
  }

  const centroids = centroidsRef.current;
  const clusterSizes = clusterSizesRef.current;

  return (
    <>
      <TopBar Message="Cluster PCA 3D Explorer" />
      <div className="cluster-explorer-container">
        <Canvas
          className="cluster-explorer-canvas"
          camera={{ position: [30, 0, -20] }}
          raycaster={{
            params: { Points: { threshold: 0.03 } },
          }}
        >
          <color attach="background" args={["#111"]} />
          <PointCloudWithHighlight
            centroids={centroids}
            clusterSizes={clusterSizes}
            selectedBucket={selectedBucket}
            setSelectedCentroid={setSelectedCentroid}
          />
          <KeyboardCameraControls
            selectedCentroid={selectedCentroid}
            centroids={centroids}
          />
        </Canvas>

        <Legend
          selectedBucket={selectedBucket}
          setSelectedBucket={setSelectedBucket}
        />

        <SideBar
          selectedCentroid={selectedCentroid}
          clusterSizes={clusterSizes}
        />
      </div>
    </>
  );
}
