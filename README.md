# YFCC100M Image Index

This project's goal is to take the raw YFCC100M corpus, run a smaller and faster object detection model over every image, and produce a structured index that makes it easy to pull out interesting subsets, which then become the bootstrap datasets HAWK searches over.

## Project Structure

```
yfcc-scope
├── archived/                                        # Obsolete code
│   ├── run_uvicorn_thread_dev_server.py
│   ├── starlett_app_3d.py
│   ├── starlett_app_with_api.py
│   ├── starlette_app.py
│   ├── yfcc_yolo_to_postgres.py
│   └── yfcc_yolo_to_postgres-shards-batch_query_metadata.py
│
├── metadata/                                        # Code for converting metadata formats
│   ├── csv2json.py
│   ├── duckdb2postgres.py
│   └── json2duckdb.py
│
├── yfcc-app/                                        # Starlette app serving API and viewers
│   ├── __init__.py
│   ├── app.py
│   ├── constants.py
│   ├── db.py
│   ├── log.py
│   ├── routes.py
│   ├── utils.py
│   ├── static/
│   │   ├── css/
│   │   │   └── app.css
│   │   └── js/
│   │       ├── freqs.js
│   │       └── page.js
│   └── templates/
│       └── index.html
│
├── yfcc-viewer                                     # Vite + React Viewer app for exploring the indexed data
│   ├── dist/                                       # Vite build output
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx                                 # Route to different viewer apps
│   │   ├── AppDashboard.jsx
│   │   ├── AppPCA3DExplorer.jsx
│   │   ├── index.css
│   │   ├── main.jsx
│   │   ├── assets/
│   │   └── image-viewer/                           # 3D Library Image Viewer app
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
├── yfcc_yolo_to_postgres-entire-shard-batch.py     # Process WebDataset shards, runs YOLO on each image, and insert results into PostgreSQL
├── LICENSE
├── README.md
└─── requirements.txt
```
