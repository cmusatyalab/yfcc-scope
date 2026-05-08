import React, { useRef, useState, useEffect } from "react";

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Layout constants ───────────────────────────────────────────────────────
const WALL_X = 440;
const IMG_W = 280;
const IMG_H = 195;
const MAT = 14;
const FRAME_W = IMG_W + MAT * 2;
const FRAME_H = IMG_H + MAT * 2;
const ROW_Y = FRAME_H / 2;

export function makeLayout(items) {
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

export default function Gallery({ items, onBack }) {
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
        Click to lock mouse · WASD to move · Q to exit
      </div>
    </div>
  );
}
