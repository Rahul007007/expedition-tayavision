import sys
import os
import modal

app = modal.App("tayavision-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    # Install everything directly into the system path
    .run_commands(
        "pip install --upgrade pip",
        "cd /root/project && pip install . vllm ray 'transformers>=4.46.0' lm-eval"
    )
    .add_local_dir("evaluation", remote_path="/root/project/evaluation", copy=True)
)

# Persistent volume for results in the cloud
results_volume = modal.Volume.from_name("tayavision-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/project/evaluation/results": results_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600 * 4,
)
def run_evaluation(task: str, model_name: str, batch_size: str = "auto", log_samples: bool = False):
    # Setup project environment inside the container
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    
    # We call your existing evaluation script as if it were running natively
    from evaluation.run_eval import main
    
    # Pass arguments to the argparse in run_eval.py
    sys.argv = [
        "run_eval.py",
        "--task", task,
        "--model-name", model_name,
        "--backend", "vllm", 
        "--batch-size", batch_size,
        "--output-dir", "/root/project/evaluation/results"
    ]
    
    if log_samples:
        sys.argv.append("--log-samples")
    
    print(f"Starting evaluation for task: {task} using model: {model_name}...")
    main()
    
    # Commit changes to the volume so they are accessible locally
    results_volume.commit()

# 3. Running our function locally and remotely
# The @app.local_entrypoint defines the starting point when we run `modal run scripts/modal_eval.py`
@app.local_entrypoint()
def main(task: str, model_name: str = "CohereLabs/tiny-aya-base", batch_size: str = "auto", log_samples: bool = False):
    import subprocess
    import shutil
    
    # We trigger the remote call that runs in the cloud with .remote()
    print("Initializing cloud GPU...")
    run_evaluation.remote(
        task=task,
        model_name=model_name,
        batch_size=batch_size,
        log_samples=log_samples
    )
    
    # Download the results from the Modal Volume directly to the local machine
    # Use a temporary folder for the raw volume download to help with flattening/prefixing
    local_results_dir = "evaluation/results"
    os.makedirs(local_results_dir, exist_ok=True)
    temp_sync_dir = ".modal_sync_temp"
    
    print(f"Syncing results from cloud...")
    try:
        # 1. Download full volume to temp dir
        subprocess.run(["modal", "volume", "get", "tayavision-results", "/", temp_sync_dir], check=True, capture_output=True)
        
        # 2. Extract and rename files from the temp dir to the final results dir
        # We un-nest them from the model-named folders and prefix them with the task
        for root, dirs, files in os.walk(temp_sync_dir):
            for filename in files:
                if filename.endswith(".json") or filename.endswith(".jsonl"):
                    # Only move files that were just generated (or match the task/timestamp logic if needed)
                    # For now, we move everything found in the volume to results/ with a task prefix
                    old_path = os.path.join(root, filename)
                    # Add task prefix if not already present
                    new_name = filename if filename.startswith(task) else f"{task}_{filename}"
                    new_path = os.path.join(local_results_dir, new_name)
                    
                    if os.path.exists(new_path):
                        os.remove(new_path)
                    shutil.move(old_path, new_path)
                    print(f"Synced and prefixed: {new_path}")
        
        # 3. Cleanup temp dir
        if os.path.exists(temp_sync_dir):
            shutil.rmtree(temp_sync_dir)
            
        print(f"\nEvaluation complete. Results are flat and prefixed in {local_results_dir}/")
    except Exception as e:
        print(f"Error syncing results: {e}")
