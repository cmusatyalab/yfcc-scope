# YFCC100M Image Index

This project's goal is to take the raw YFCC100M corpus and produce structured indexes that make it easy to pull out interesting subsets with natural language queries, which then become the datasets HAWK searches over.

## Installation

You need to have [uv](https://docs.astral.sh/uv) and [npm](https://nodejs.org) installed.

```
# setup .venv virtual environment / install Python dependencies
uv sync

# build Vite frontend code (/yfcc-viewer)
uv build

# run Python backend (/src/yfcc_scope)
uv run uvicorn yfcc_scope.app:app --host 0.0.0.0 --port 8080
```

## Project Structure

```
yfcc-scope
├── archived/                           # Obsolete code
│   ├── run_uvicorn_thread_dev_server.py
│   ├── starlett_app_3d.py
│   ├── starlett_app_with_api.py
│   ├── starlette_app.py
│   ├── yfcc_yolo_to_postgres.py
│   └── yfcc_yolo_to_postgres-shards-batch_query_metadata.py
│
├── clip-embedding/                     # Compute CLIP embeddings and insert into PostgreSQL
│   ├── clip_to_postgres.py
│   ├── yfcc_image_embeddings.npy
│   ├── yfcc_img_to_clip.ipynb
│   └── yfcc_img_to_clip.py
│
├── metadata/                           # Code for converting metadata formats
│   ├── csv2json.py
│   ├── duckdb2postgres.py
│   └── json2duckdb.py
│
├── yolo-to-postgres/                   # WebDataset fetch - YOLO detection - PostgreSQL insertion pipeline
│   ├── yfcc_yolo_to_postgres-entire-shard-batch.py
│   └── missing_images.ipynb
│
├── src/yfcc_scope/                           # Starlette app serving API and viewers
│   ├── __init__.py
│   ├── app.py
│   ├── constants.py
│   ├── db.py
│   ├── log.py
│   ├── routes.py
│   ├── utils.py
│   ├── settings.py
│   ├── static/
│   │   ├── css/
│   │   │   └── app.css
│   │   └── js/
│   │       ├── freqs.js
│   │       └── page.js
│   └── templates/
│       └── index.html
│
├── yfcc-viewer/                       # Vite + React Viewer app for exploring the indexed data
│   ├── dist/                          # Vite build output
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx                    # Route to different viewer apps
│   │   ├── AppDashboard.jsx
│   │   ├── AppPCA3DExplorer.jsx
│   │   ├── index.css
│   │   ├── main.jsx
│   │   ├── assets/
│   │   └── image-viewer/              # 3D Library Image Viewer app
│   │       ├── AppImageViewer.css
│   │       ├── AppImageViewer.jsx
│   │       ├── Gallery.jsx
│   │       ├── ImageResultsPanel.jsx
│   │       ├── SearchControlPanel.jsx
│   │       ├── SelectedPanel.jsx
│   │       ├── SqlDisplayPanel.jsx
│   │       └── sqlPrompt.js
│   │
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   ├── README.md
│   └── vite.config.js
│
├── LICENSE
├── README.md
├── dotenv.example			# example .env configuration file
├── pyproject.toml			# python dependencies and packaging management
└── hatch_build.py			# build hooks for the frontend components
```
