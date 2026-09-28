import React, { useEffect, useState } from "react";
import ImageUpload from "../components/ImageUpload";
import Slider from "../components/Slider";
import { getErrorMessage } from "../utils";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const API_PREFIX = `${API_BASE}/api`;

export default function ImageNearestCentroids({
  embeddingType,
  nearestCentroidIds,
  setNearestCentroidIds,
}) {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [nearestCentroidCount, setNearestCentroidCount] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setNearestCentroidIds([]);
    setError("");
  }, [uploadedImage, embeddingType]);

  const searchNearestCentroids = async () => {
    if (!uploadedImage || isLoading) return;

    setIsLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", uploadedImage);
      formData.append("limit", String(nearestCentroidCount));

      const response = await fetch(
        `${API_PREFIX}/image_nearest_centroids?embedding=${embeddingType}`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const data = await response.json();
      setNearestCentroidIds(data.row_ids || []);
    } catch (requestError) {
      setNearestCentroidIds([]);
      setError(
        requestError instanceof Error
          ? requestError.message
          : String(requestError),
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="cluster-explorer-options">
      <div className="cluster-explorer-image-title-row">
        <div className="cluster-explorer-options-title">
          Image Nearest Centroids
        </div>
        <button
          type="button"
          className="cluster-explorer-centroid-submit"
          onClick={searchNearestCentroids}
          disabled={!uploadedImage || isLoading}
        >
          {isLoading ? "Searching..." : "Search"}
        </button>
      </div>
      <ImageUpload file={uploadedImage} setFile={setUploadedImage} />
      <Slider
        label="Number of nearest centroids"
        value={nearestCentroidCount}
        min={1}
        max={20}
        onChange={setNearestCentroidCount}
      />
      {error ? (
        <div className="cluster-explorer-error-message" role="alert">
          {error}
        </div>
      ) : null}
      {nearestCentroidIds.length > 0 ? (
        <div className="cluster-explorer-nearest-results">
          Row IDs: {nearestCentroidIds.join(", ")}
        </div>
      ) : null}
    </div>
  );
}
