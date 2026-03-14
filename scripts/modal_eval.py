import sys
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
def run_evaluation(task: str, model_name: str, batch_size: str = "auto"):
    import os
    
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
    
    print(f"Starting evaluation for task: {task} using model: {model_name}...")
    main()
    
    # Read generated results to send them back locally
    results_data = {}
    output_dir = "/root/project/evaluation/results"
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            with open(os.path.join(output_dir, filename), "r") as f:
                results_data[filename] = f.read()
    
    results_volume.commit()
    return results_data

# 3. Running our function locally and remotely
# The @app.local_entrypoint defines the starting point when we run `modal run scripts/modal_eval.py`
@app.local_entrypoint()
def main(task: str, model_name: str = "CohereLabs/tiny-aya-base", batch_size: str = "auto"):
    
    # We trigger the remote call that runs in the cloud with .remote()
    print("Initializing cloud GPU...")
    results_dict = run_evaluation.remote(
        task=task,
        model_name=model_name,
        batch_size=batch_size
    )
    
    # Write the results back to the local results directory
    local_results_dir = "evaluation/results"
    os.makedirs(local_results_dir, exist_ok=True)
    
    for filename, content in results_dict.items():
        local_path = os.path.join(local_results_dir, filename)
        with open(local_path, "w") as f:
            f.write(content)
        print(f"Synced result: {local_path}")
    
    print("\nEvaluation complete. Results are stored in evaluation/results/")
