import React from "react";
import "./Slider.css";

export default function Slider({ label, value, min, max, onChange }) {
  return (
    <label className="slider-control">
      <span className="slider-label">{label}</span>
      <input
        className="slider-input"
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="slider-value">{value}</span>
    </label>
  );
}