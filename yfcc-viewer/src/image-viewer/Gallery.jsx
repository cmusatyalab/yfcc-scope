import React, { useRef, useState, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { PointerLockControls, Html, Text } from "@react-three/drei";
import * as THREE from "three";
import { rowKey } from "./ImageResultsPanel";

// ─── Layout constants ───────────────────────────────────────────────────────
const WALL_X = 440;
const IMG_W = 280;
const IMG_H = 195;
const MAT = 14;
const FRAME_W = IMG_W + MAT * 2;
const FRAME_H = IMG_H + MAT * 2;
const ROW_Y = FRAME_H / 2;
const FRAME_Y_OFFSET = 32;
const IMG_SCALE = 36;

export function makeLayout(items) {
  return items.map((item, i) => {
    const col = Math.floor(i / 4);
    const slot = i % 4;
    return {
      ...item,
      side: slot % 2 === 0 ? "left" : "right",
      wy: slot < 2 ? -ROW_Y + FRAME_Y_OFFSET : ROW_Y + FRAME_Y_OFFSET,
      wz: -(col * FRAME_W + 400),
    };
  });
}

// Custom Player component for WASD navigation
function Player({ onBack, onShowSelected, floorLen }) {
  const keys = useRef({});
  const speed = 800;
  const direction = new THREE.Vector3();
  const frontVector = new THREE.Vector3();
  const sideVector = new THREE.Vector3();
  const { gl } = useThree();

  useEffect(() => {
    const handleKeyDown = (e) => {
      keys.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "q") onBack();
      if (e.key.toLowerCase() === "e") {
        document.exitPointerLock();
        onShowSelected();
      }
    };
    const handleKeyUp = (e) => {
      keys.current[e.key.toLowerCase()] = false;
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [onBack, onShowSelected, gl]);

  useFrame((state, delta) => {
    const k = keys.current;

    // Movement axes based on camera orientation
    frontVector.set(
      0,
      0,
      (k["s"] || k["arrowdown"] ? 1 : 0) - (k["w"] || k["arrowup"] ? 1 : 0),
    );
    sideVector.set(
      (k["d"] || k["arrowright"] ? 1 : 0) - (k["a"] || k["arrowleft"] ? 1 : 0),
      0,
      0,
    );

    direction
      .subVectors(frontVector, sideVector)
      .normalize()
      .multiplyScalar(speed * delta);

    // Apply movement
    state.camera.translateX(-direction.x);
    state.camera.translateZ(direction.z);

    // Clamping to avoid walking through walls (simulated padding)
    const limitX = WALL_X - 100;
    state.camera.position.x = THREE.MathUtils.clamp(
      state.camera.position.x,
      -limitX,
      limitX,
    );
    // Do not walk backward past origin or forward past floor end
    state.camera.position.z = THREE.MathUtils.clamp(
      state.camera.position.z,
      -(floorLen || 1000) + 100,
      -100,
    );

    // Keep eye level locked
    state.camera.position.y = 0;
  });

  return null;
}

// Renders an individual image frame
function Frame({ item, index, onClick, isSelected }) {
  const isLeft = item.side === "left";
  const x = isLeft ? -(WALL_X - 6) : WALL_X - 6;
  const y = item.wy;
  const z = item.wz;
  const rotY = isLeft ? Math.PI / 2 : -Math.PI / 2;

  const src = item.thumb_url || item.path || "";
  const [hovered, setHovered] = useState(false);
  const [inView, setInView] = useState(false);
  const inViewRef = useRef(false);

  useFrame((state) => {
    // Determine distance from camera
    const dist = Math.abs(state.camera.position.z - z);
    // Load if within 3000 units (~3-4 columns away)
    const shouldBeInView = dist < 3000;
    if (shouldBeInView !== inViewRef.current) {
      inViewRef.current = shouldBeInView;
      setInView(shouldBeInView);
    }
  });

  // Outline color based on selection and hover
  const frameColor = isSelected ? "#2466a4" : hovered ? "#555" : "#222";

  return (
    <group position={[x, y, z]} rotation={[0, rotY, 0]}>
      {/* Interactive Backboard */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onClick(item, index);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          // document.body.style.cursor = "pointer";
          setHovered(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          // document.body.style.cursor = "auto";
          setHovered(false);
        }}
      >
        <planeGeometry args={[FRAME_W, FRAME_H]} />
        <meshBasicMaterial color={frameColor} />
      </mesh>

      {src && inView ? (
        /* Image display (Using HTML overlay to bypass WebGL CORS restrictions) */
        <Html
          transform
          position={[0, 4, 1]} // Extrude slightly to avoid z-fighting
          style={{ pointerEvents: "none" }} // Let the mesh underneath handle clicks
        >
          <img
            src={src}
            alt=""
            style={{
              width: `${IMG_W * IMG_SCALE}px`,
              height: `${IMG_H * IMG_SCALE}px`,
              objectFit: "cover",
              display: "block",
            }}
            draggable={false}
          />
        </Html>
      ) : (
        /* Placeholder for unloaded images */
        <mesh position={[0, 0, 1]} raycast={() => null}>
          <planeGeometry args={[IMG_W, IMG_H]} />
          <meshBasicMaterial color="#333" />
        </mesh>
      )}

      {/* Caption beneath frame */}
      <Text
        position={[0, -(IMG_H / 2) + 8, 2]}
        fontSize={12}
        color="white"
        anchorX="center"
        anchorY="top"
        raycast={() => null}
      >
        {item.image_file_id || "—"}
      </Text>
    </group>
  );
}

export default function Gallery({
  items,
  onBack,
  searchResults,
  selectedIds,
  toggleSelected,
  onDownloadSelected,
  downloading,
  showSelectedPanel,
  setShowSelectedPanel,
}) {
  const cols = Math.ceil(items.length / 4) || 1;
  const floorLen = cols * FRAME_W + 400;
  const limitX = WALL_X;

  const handleShowSelected = () => {
    setShowSelectedPanel(true);
  };

  return (
    <div className="gallery-root">
      {/* Center crosshair */}
      <div className="gallery-crosshair" />

      <Canvas camera={{ position: [0, 0, 0], fov: 75, near: 0.1, far: 20000 }}>
        {/* Lights */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[0, 10, 0]} intensity={1.5} />

        {/* First Person Controls */}
        <Player
          onBack={onBack}
          onShowSelected={handleShowSelected}
          floorLen={floorLen}
        />
        {!showSelectedPanel && <PointerLockControls />}

        {/* Floor */}
        <mesh
          position={[0, -200, -floorLen / 2]}
          rotation={[-Math.PI / 2, 0, 0]}
        >
          <planeGeometry args={[limitX * 2, floorLen]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>

        {/* Ceiling */}
        <mesh position={[0, 300, -floorLen / 2]} rotation={[Math.PI / 2, 0, 0]}>
          <planeGeometry args={[limitX * 2, floorLen]} />
          <meshStandardMaterial color="#0a0a0a" />
        </mesh>

        {/* Left Wall */}
        <mesh
          position={[-limitX, 0, -floorLen / 2]}
          rotation={[0, Math.PI / 2, 0]}
        >
          <planeGeometry args={[floorLen, 1000]} />
          <meshStandardMaterial color="#222" />
        </mesh>

        {/* Right Wall */}
        <mesh
          position={[limitX, 0, -floorLen / 2]}
          rotation={[0, -Math.PI / 2, 0]}
        >
          <planeGeometry args={[floorLen, 1000]} />
          <meshStandardMaterial color="#222" />
        </mesh>

        {/* Images */}
        {items.map((item, i) => (
          <Frame
            key={rowKey(item, i)}
            index={i}
            item={item}
            isSelected={selectedIds.has(rowKey(item, i))}
            onClick={(selectedItem, index) => {
              toggleSelected(selectedItem, index);
            }}
          />
        ))}
      </Canvas>

      {/* HUD Info */}
      <div className="gallery-hud">
        Move mouse to look around · WASD to move · Click image to select · E to
        see selection · Q to exit
      </div>
    </div>
  );
}
