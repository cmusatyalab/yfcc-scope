import React, { useEffect, useRef, useState } from "react";
import "./ImageUpload.css";

export default function ImageUpload({ file, setFile }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    const handlePaste = (event) => {
      for (const item of event.clipboardData?.items || []) {
        if (item.type.startsWith("image/")) {
          setFile(item.getAsFile());
          break;
        }
      }
    };

    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [setFile]);

  const showFile = (nextFile) => {
    if (!nextFile || !nextFile.type.startsWith("image/")) return;
    setFile(nextFile);
  };

  const clearFile = () => {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dropZoneRef.current?.classList.remove("drag-over");
    showFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="image-upload">
      {!file && (
        <div
          className="upload-zone"
          ref={dropZoneRef}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            dropZoneRef.current?.classList.add("drag-over");
          }}
          onDragLeave={() => dropZoneRef.current?.classList.remove("drag-over")}
          onDrop={handleDrop}
        >
          <p className="upload-title">
            Drag &amp; drop / Paste from clipboard / Click to browse
          </p>
          <p className="upload-sub">Supported formats: PNG, JPG, WebP</p>
        </div>
      )}

      {file && previewUrl && (
        <div className="preview-area">
          <img className="preview-thumb" src={previewUrl} alt="preview" />
          <div className="preview-meta">
            <p className="preview-name">{file.name || ""}</p>
            <p className="preview-size">
              {(file.size / 1024).toFixed(0)} KB · {file.type}
            </p>
          </div>
          <button type="button" className="preview-clear" onClick={clearFile}>
            Clear
          </button>
        </div>
      )}

      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(event) => showFile(event.target.files[0])}
      />
    </div>
  );
}
