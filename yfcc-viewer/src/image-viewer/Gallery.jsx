import React, { useRef, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { PointerLockControls, Html, Text } from "@react-three/drei";
import * as THREE from "three";

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
function Player({ onBack }) {
  const keys = useRef({});
  const speed = 800;
  const direction = new THREE.Vector3();
  const frontVector = new THREE.Vector3();
  const sideVector = new THREE.Vector3();

  useEffect(() => {
    const handleKeyDown = (e) => {
      keys.current[e.key.toLowerCase()] = true;
      if (e.key.toLowerCase() === "q") onBack();
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
  }, [onBack]);

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
    // Do not walk backward past origin
    state.camera.position.z = Math.min(100, state.camera.position.z);

    // Keep eye level locked
    state.camera.position.y = 0;
  });

  return null;
}

// Renders an individual image frame
function Frame({ item, onClick }) {
  const isLeft = item.side === "left";
  const x = isLeft ? -(WALL_X - 6) : WALL_X - 6;
  const y = item.wy;
  const z = item.wz;
  const rotY = isLeft ? Math.PI / 2 : -Math.PI / 2;

  const src = item.thumb_url || item.path || "";
  const [hovered, setHovered] = useState(false);

  return (
    <group position={[x, y, z]} rotation={[0, rotY, 0]}>
      {/* Interactive Backboard */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onClick(item);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          setHovered(false);
        }}
      >
        <planeGeometry args={[FRAME_W, FRAME_H]} />
        <meshBasicMaterial color={hovered ? "#555" : "#222"} />
      </mesh>

      {/* Image display (Using HTML overlay to bypass WebGL CORS restrictions) */}
      {src && (
        <Html
          transform
          position={[0, 4, 1]} // Extrude slightly to avoid z-fighting
          style={{ pointerEvents: "none" }} // Let the mesh underneath handle clicks
        >
          <div
            style={{
              width: `${IMG_W * IMG_SCALE}px`,
              height: `${IMG_H * IMG_SCALE}px`,
            }}
          >
            <img
              src={src}
              alt=""
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                display: "block",
              }}
              draggable={false}
            />
          </div>
        </Html>
      )}

      {/* Caption beneath frame */}
      <Text
        position={[0, -(IMG_H / 2) + 8, 2]}
        fontSize={12}
        color="white"
        anchorX="center"
        anchorY="top"
      >
        {item.image_file_id || "—"}
      </Text>
    </group>
  );
}

export default function Gallery({ items, onBack }) {
  const cols = Math.ceil(items.length / 4) || 1;
  const floorLen = cols * FRAME_W * 2 + 4000;
  const limitX = WALL_X;

  return (
    <div className="gallery-root">
      {/* Center crosshair */}
      <div className="gallery-crosshair" />

      <Canvas camera={{ position: [0, 0, 0], fov: 75, near: 0.1, far: 20000 }}>
        {/* Lights */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[0, 10, 0]} intensity={1.5} />

        {/* First Person Controls */}
        <Player onBack={onBack} />
        <PointerLockControls />

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
            key={i}
            item={item}
            onClick={(selectedItem) => {
              // Add selection logic here later
              console.log("Selected Image ID:", selectedItem.image_file_id);
              alert("Selected Image: " + selectedItem.image_file_id);
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
