import os
import time
import uuid
import shutil
import json
import csv
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from main import run_pipeline
from src.config import load_config

import sys

# Force immediate unbuffered stdout/stderr for real-time live logs on Google Colab
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

app = FastAPI(title="Statcut Analytics - Football Match Analysis API (18-Stage Pipeline)")

# Allow all origins for seamless Ngrok and local dev connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("data/input_videos", exist_ok=True)
os.makedirs("data/output_videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static media directories
app.mount("/media/videos", StaticFiles(directory="data/output_videos"), name="output_videos")
app.mount("/media/reports", StaticFiles(directory="reports"), name="reports")

templates = Jinja2Templates(directory="templates")

# Global in-memory storage for video processing jobs
JOBS: Dict[str, Dict[str, Any]] = {}

def async_pipeline_worker(job_id: str, input_path: str, output_path: str):
    """
    Executes the 18-stage Football Analytics pipeline in a background worker.
    Updates job status, progress percentage, and accumulates real-time log entries.
    """
    try:
        def update_progress(stage_name: str, progress_pct: int):
            t_stamp = time.strftime('%H:%M:%S')
            log_line = f"[{t_stamp}] [{progress_pct}%] {stage_name}"
            
            existing_logs = JOBS.get(job_id, {}).get("logs", [])
            if not existing_logs or existing_logs[-1] != log_line:
                existing_logs.append(log_line)

            JOBS[job_id] = {
                "status": "processing",
                "progress": progress_pct,
                "stage": stage_name,
                "input_file": input_path,
                "logs": existing_logs
            }

        # Initial state
        update_progress("Video Ingestion & Frame Extraction", 5)
        
        # Execute 18-stage pipeline without stale stubs
        run_pipeline(
            input_video_path=input_path,
            output_video_path=output_path,
            use_stubs=False,
            progress_callback=update_progress
        )

        output_filename = os.path.basename(output_path)
        existing_logs = JOBS.get(job_id, {}).get("logs", [])
        existing_logs.append(f"[{time.strftime('%H:%M:%S')}] [100%] Analysis Complete (18-Stage Pipeline)")

        JOBS[job_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Analysis Complete (18-Stage Pipeline)",
            "logs": existing_logs,
            "output_video_url": f"/media/videos/{output_filename}",
            "heatmap_url": "/media/reports/heatmap.png",
            "heatmap_team_a_url": "/media/reports/heatmap_team_a.png",
            "heatmap_team_b_url": "/media/reports/heatmap_team_b.png",
            "dashboard_url": "/media/reports/dashboard.html",
            "tactical_report_url": "/media/reports/tactical_report.txt",
            "tactical_data_url": "/media/reports/tactical_data.json"
        }
    except Exception as e:
        import traceback
        print(f"Job {job_id} failed: {e}")
        traceback.print_exc()
        existing_logs = JOBS.get(job_id, {}).get("logs", [])
        existing_logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {str(e)}")
        JOBS[job_id] = {
            "status": "failed",
            "progress": 0,
            "stage": f"Error: {str(e)}",
            "logs": existing_logs
        }

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Serves the Premier League / Champions League Dark Sports Analytics Web UI.
    """
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to upload match video clip (.mp4, .avi, .mov) and launch 18-stage analysis pipeline.
    """
    allowed_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".ts", ".3gp"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Use .mp4, .avi, or .mov")

    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join("data", "input_videos", f"upload_{job_id}{ext}")
    output_path = os.path.join("data", "output_videos", f"match_{job_id}_annotated.mp4")

    with open(input_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    init_log = f"[{time.strftime('%H:%M:%S')}] [5%] Video Uploaded: {file.filename}. Queued for 18-stage analysis..."
    JOBS[job_id] = {
        "status": "queued",
        "progress": 5,
        "stage": "Uploaded. Queued for 18-stage analysis...",
        "filename": file.filename,
        "logs": [init_log]
    }

    background_tasks.add_task(async_pipeline_worker, job_id, input_path, output_path)

    return JSONResponse({
        "status": "success",
        "job_id": job_id,
        "message": "Video uploaded successfully. 18-stage analysis pipeline started."
    })

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Returns real-time processing status and stage progress percentage.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS[job_id]

@app.get("/api/download/{filename}")
async def download_video(filename: str):
    """
    Direct force-download endpoint with Content-Disposition attachment header.
    """
    file_path = os.path.join("data", "output_videos", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Annotated video '{filename}' not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    """
    Returns comprehensive match statistics, player stats, timeline events,
    tactical data, and visualization URLs.
    """
    # Player stats
    player_stats_path = os.path.join("reports", "stats_player.csv")
    players = []
    if os.path.exists(player_stats_path):
        with open(player_stats_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                players.append({
                    "id": row.get("Player_ID"),
                    "jersey": row.get("Jersey_Number", row.get("Player_ID")),
                    "team_id": int(row.get("Team_ID", 0)),
                    "team": row.get("Team_Name"),
                    "distance_m": float(row.get("Total_Distance_Meters", 0)),
                    "avg_speed": float(row.get("Avg_Speed_km_h", 0)),
                    "max_speed": float(row.get("Max_Speed_km_h", 0)),
                    "sprint_count": int(row.get("Sprint_Count", 0)),
                    "touch_count": int(row.get("Touch_Count", 0)),
                    "pass_count": int(row.get("Pass_Count", 0)),
                    "avg_acceleration": float(row.get("Avg_Acceleration_ms2", 0)),
                })

    # Events
    events = []
    events_path = os.path.join("reports", "events.csv")
    if os.path.exists(events_path):
        with open(events_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append({
                    "timestamp": row.get("Timestamp", ""),
                    "event_type": row.get("Event_Type", ""),
                    "confidence": float(row.get("Confidence", 0)),
                    "description": row.get("Description", "")
                })

    # Tactical data
    tactical = {}
    tactical_path = os.path.join("reports", "tactical_data.json")
    if os.path.exists(tactical_path):
        with open(tactical_path, "r", encoding="utf-8") as f:
            tactical = json.load(f)

    # Summary statistics (possession, xG, pass accuracy, goals, stage metrics)
    summary = {}
    summary_path = os.path.join("reports", "summary_stats.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    possession = summary.get('possession', {"home": 50.0, "away": 50.0})
    xg = summary.get('xg', {"home": 0.0, "away": 0.0})
    pass_accuracy = summary.get('pass_accuracy', {"home": 80.0, "away": 80.0})
    goals = summary.get('goals', {"home": 0, "away": 0})
    stage_metrics = summary.get('stage_metrics', {})

    job_info = JOBS.get(job_id, {})
    return {
        "job_id": job_id,
        "status": job_info.get("status", "completed"),
        "video_url": job_info.get("output_video_url", "/media/videos/match_01_annotated.mp4"),
        "heatmap_url": job_info.get("heatmap_url", "/media/reports/heatmap.png"),
        "heatmap_team_a_url": job_info.get("heatmap_team_a_url", "/media/reports/heatmap_team_a.png"),
        "heatmap_team_b_url": job_info.get("heatmap_team_b_url", "/media/reports/heatmap_team_b.png"),
        "dashboard_url": job_info.get("dashboard_url", "/media/reports/dashboard.html"),
        "tactical_data": tactical,
        "players": players,
        "events": events,
        "possession": possession,
        "xg": xg,
        "pass_accuracy": pass_accuracy,
        "goals": goals,
        "stage_metrics": stage_metrics
    }

@app.delete("/api/match/{job_id}")
async def delete_match(job_id: str):
    """
    Deletes match job record and associated video/report deliverables.
    """
    if job_id in JOBS:
        job = JOBS.pop(job_id)
        # Clean input file if exists
        inp = job.get("input_file")
        if inp and os.path.exists(inp) and "default" not in inp:
            try: os.remove(inp)
            except: pass
        # Clean output video if exists
        out = job.get("output_file")
        if out and os.path.exists(out):
            try: os.remove(out)
            except: pass

    return {"status": "success", "message": f"Match {job_id} and analytics deleted successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
