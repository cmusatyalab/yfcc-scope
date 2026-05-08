// Main app component for YFCC viewer. Sets up the top-level layout and routing for the different views (image viewer, dashboard, PCA 3D explorer).

import React from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import Dashboard from "./AppDashboard.jsx";
import PCA3DExplorer from "./AppPCA3DExplorer.jsx";
import ImageViewer from "./image-viewer/AppImageViewer.jsx";
import "./App.css";

export default function App() {
  const linkClassName = ({ isActive }) =>
    `app-nav-link${isActive ? " active" : ""}`;

  return (
    <div className="app-shell">
      <div className="app-topbar">
        <div className="app-topbar-inner">
          <div className="app-brand">YFCC Viewer</div>
          <nav className="app-nav">
            <NavLink to="/image" className={linkClassName}>
              Image Viewer
            </NavLink>
            <NavLink to="/dashboard" className={linkClassName}>
              Dashboard
            </NavLink>
            <NavLink to="/pca3d" className={linkClassName}>
              3D PCA Viewer
            </NavLink>
          </nav>
        </div>
      </div>

      <Routes>
        <Route path="/" element={<Navigate to="/image" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/pca3d" element={<PCA3DExplorer />} />
        <Route path="/image" element={<ImageViewer />} />
        <Route path="*" element={<Navigate to="/image" replace />} />
      </Routes>
    </div>
  );
}
