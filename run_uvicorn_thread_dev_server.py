import importlib, threading, uvicorn, sys

import starlette_app
importlib.reload(starlette_app)         
app = starlette_app.app                 

# use a new port to avoid the old thread
PORT = 8060

def run():
    uvicorn.run(app, host="127.0.0.1", port=PORT, reload=False)

t = threading.Thread(target=run, daemon=True)
t.start()

from IPython.display import IFrame
IFrame(src=f"v", width="100%", height=750)