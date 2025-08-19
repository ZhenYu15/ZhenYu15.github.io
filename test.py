# server.py
import uvicorn
from fastapi import FastAPI, Body, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import torch

app = FastAPI()

# Allow frontend (index.html) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "model_code": None,
    "model": None,
    "example_input": None,
    "pth_path": None,
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_user_model_from_code(code: str):
    """Exec user code safely and return (model, example_input)."""
    g = {}
    try:
        exec(code, g)
    except Exception as e:
        raise RuntimeError(f"Code execution error: {e}")

    if "build_model" not in g:
        raise RuntimeError("You must define a build_model() function.")

    model = g["build_model"]()
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("build_model() must return nn.Module.")

    if "build_example_input" in g:
        example_input = g["build_example_input"](model)
    else:
        # fallback: one dummy input
        example_input = torch.randn(1, 1)

    return model, example_input

# --------------------------------------------------
# API
# --------------------------------------------------
@app.post("/api/set_code")
async def set_code(payload: Dict[str, str] = Body(...)):
    code = payload.get("code", "").strip()
    if not code:
        return JSONResponse({"ok": False, "error": "Empty code"}, status_code=400)

    try:
        model, example_input = load_user_model_from_code(code)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    STATE.update({
        "model_code": code,
        "model": model,
        "example_input": example_input
    })
    return {"ok": True, "info": "Model code accepted."}

@app.post("/api/upload_pth")
async def upload_pth(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    STATE["pth_path"] = path
    return {"ok": True, "info": f"File saved at {path}"}

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
