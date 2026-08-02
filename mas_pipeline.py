"""
Autonomous Multi-Agent System (MAS) for Neural Network Curation, Training & Deployment
========================================================================================

A LangGraph state machine that orchestrates 6 specialized agents:
  1. Data Curator      – Generates synthetic datasets
  2. Systems Architect – Writes PyTorch training scripts via Gemini
  3. Executor          – Runs the training script in a subprocess
  4. Critic            – Evaluates execution logs and decides retry/success
  5. MLOps Engineer    – Generates a FastAPI deployment server
  6. HF Publisher      – Creates a Hugging Face repo and uploads all artifacts

Run in a Kaggle notebook with a free GPU runtime for best results.
"""



# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import subprocess
import textwrap
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# ──────────────────────────────────────────────────────────────────────────────
# 2. State Definition
# ──────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state that flows through every node in the graph."""
    user_request: str        # The original user prompt describing the task
    dataset_path: str        # Filesystem path to the generated dataset
    model_code: str          # Full Python source of the training script
    execution_logs: str      # Combined stdout + stderr from training
    deployment_package: str  # Full Python source of the FastAPI server
    iteration_count: int     # Number of architect→executor→critic cycles
    hf_repo_url: str         # Hugging Face repo URL after upload


# ──────────────────────────────────────────────────────────────────────────────
# 3. LLM Factory
# ──────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance pointed at a capable model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=temperature,
        max_output_tokens=8192,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Agent Nodes
# ──────────────────────────────────────────────────────────────────────────────

# ── Node 1: Data Curator ─────────────────────────────────────────────────────

def data_curator(state: GraphState) -> dict:
    """
    Generate a synthetic image dataset on disk.

    Uses the Gemini LLM to write a Python script that creates a local folder
    of synthetic images (random tensors saved as PNGs with class-based
    subdirectories).  The script is then executed so the dataset exists on
    disk for downstream training.
    """
    llm = _get_llm()
    user_request = state["user_request"]

    prompt = textwrap.dedent(f"""\
        You are a data-engineering agent.  The user wants to build a model for
        the following task:

        >>> {user_request} <<<

        Write a *complete* Python script (no markdown fences, no explanation)
        that creates a synthetic image dataset at the path "./synthetic_dataset".

        Requirements:
        - Use Pillow to generate random-colored 32×32 images.
        - Create 3 class subdirectories (class_0, class_1, class_2).
        - Generate 50 images per class (150 total).
        - Print "Dataset created at ./synthetic_dataset" when done.
        - The script must be self-contained (import everything it needs).
    """)

    response = llm.invoke([
        SystemMessage(content="You are a senior data engineer. Output ONLY executable Python code, nothing else."),
        HumanMessage(content=prompt),
    ])

    dataset_script = response.content.strip()
    # Strip markdown fences if the LLM wraps them
    if dataset_script.startswith("```"):
        lines = dataset_script.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        dataset_script = "\n".join(lines)

    # Write and execute the data-generation script
    script_path = "generate_dataset.py"
    with open(script_path, "w") as f:
        f.write(dataset_script)

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"[DataCurator] ⚠ Dataset script stderr:\n{result.stderr}")
    else:
        print(f"[DataCurator] ✓ {result.stdout.strip()}")

    return {
        "dataset_path": "./synthetic_dataset",
    }


# ── Node 2: Systems Architect ────────────────────────────────────────────────

def systems_architect(state: GraphState) -> dict:
    """
    Use Gemini to write a complete PyTorch training script.

    If previous execution logs contain errors (from a retry loop), they are
    included in the prompt so the LLM can fix the issues.
    """
    llm = _get_llm(temperature=0.3)
    user_request = state["user_request"]
    dataset_path = state["dataset_path"]
    previous_logs = state.get("execution_logs", "")

    error_context = ""
    if previous_logs and "SUCCESS" not in previous_logs[:200]:
        error_context = textwrap.dedent(f"""\
            ⚠ The previous training run FAILED with these logs:
            ---
            {previous_logs[-3000:]}
            ---
            You MUST fix the errors described above in your new script.
        """)

    prompt = textwrap.dedent(f"""\
        You are a senior ML systems architect.

        USER TASK: {user_request}
        DATASET PATH: {dataset_path}
        (The dataset has subdirectories class_0, class_1, class_2 with 32×32 PNG images.)

        {error_context}

        Write a COMPLETE, SELF-CONTAINED Python training script called `train.py`.

        Hard requirements:
        1. Use PyTorch and torchvision.
        2. Load the dataset from "{dataset_path}" using
           torchvision.datasets.ImageFolder and a DataLoader.
        3. Define a simple CNN (Conv2d → ReLU → MaxPool → Flatten → Linear).
           The CNN must accept 3-channel 32×32 images and output 3 classes.
        4. Use CrossEntropyLoss and Adam optimizer (lr=1e-3).
        5. Train for exactly 5 epochs.
        6. After each epoch, print: "Epoch {{epoch}}/5 — Loss: {{avg_loss:.4f}}"
        7. After training, save the model weights to "best_model.pt" using
           torch.save(model.state_dict(), "best_model.pt").
        8. Print "Training complete. Model saved to best_model.pt" at the end.
        9. Handle the case where CUDA is not available by falling back to CPU.
        10. Use torchvision.transforms to Resize(32), ToTensor(), and Normalize.

        Output ONLY the Python code. No markdown fences, no commentary.
    """)

    response = llm.invoke([
        SystemMessage(content="You are a senior ML engineer. Output ONLY valid, executable Python code."),
        HumanMessage(content=prompt),
    ])

    model_code = response.content.strip()
    if model_code.startswith("```"):
        lines = model_code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        model_code = "\n".join(lines)

    print(f"[SystemsArchitect] ✓ Generated train.py ({len(model_code)} chars)")
    return {
        "model_code": model_code,
    }


# ── Node 3: Executor ─────────────────────────────────────────────────────────

def executor(state: GraphState) -> dict:
    """
    Save the model_code to `train.py` and run it in a subprocess.
    Captures stdout and stderr into execution_logs.
    """
    model_code = state["model_code"]
    script_path = "train.py"

    with open(script_path, "w") as f:
        f.write(model_code)

    print("[Executor] ▶ Running train.py …")
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout for training
        )
        logs = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        logs = "ERROR: Training script timed out after 600 seconds."
        exit_code = -1

    status = "✓ success" if exit_code == 0 else f"✗ exit code {exit_code}"
    print(f"[Executor] {status}")

    return {
        "execution_logs": logs,
    }


# ── Node 4: Critic ───────────────────────────────────────────────────────────

def critic(state: GraphState) -> dict:
    """
    Evaluate the execution logs using Gemini.

    Returns a verdict:
      - "SUCCESS" if training completed without errors
      - A detailed error explanation otherwise
    """
    llm = _get_llm(temperature=0.0)
    execution_logs = state["execution_logs"]
    iteration_count = state.get("iteration_count", 0)

    prompt = textwrap.dedent(f"""\
        You are a senior ML code reviewer.  Analyze the following execution
        logs from a PyTorch training script.

        LOGS:
        ---
        {execution_logs[-4000:]}
        ---

        Rules:
        - If the training completed successfully (model saved, no crashes,
          loss is a finite number), respond with EXACTLY the word: SUCCESS
        - If there is a RuntimeError, ImportError, SyntaxError, or the loss
          is NaN/Inf, respond with a concise explanation of the root cause
          and a specific suggestion for how to fix the code.

        Respond with ONLY "SUCCESS" or your error analysis. Nothing else.
    """)

    response = llm.invoke([
        SystemMessage(content="You are a precise code reviewer. Be concise."),
        HumanMessage(content=prompt),
    ])

    verdict = response.content.strip()
    new_iteration = iteration_count + 1

    if "SUCCESS" in verdict.upper():
        print(f"[Critic] ✓ Training succeeded on iteration {new_iteration}")
    else:
        print(f"[Critic] ✗ Found issues (iteration {new_iteration}/3):\n  {verdict[:200]}")

    return {
        "execution_logs": verdict,  # Overwrite with the critic's summary
        "iteration_count": new_iteration,
    }


# ── Node 5: MLOps Engineer ───────────────────────────────────────────────────

def mlops_engineer(state: GraphState) -> dict:
    """
    Generate a FastAPI app that serves the trained model as a REST endpoint.
    """
    llm = _get_llm(temperature=0.2)
    user_request = state["user_request"]
    model_code = state["model_code"]

    # Extract the model class definition from model_code for reuse
    prompt = textwrap.dedent(f"""\
        You are a senior MLOps engineer.

        The user trained a PyTorch model for: {user_request}

        Here is the training script that was used (you need the model class
        definition from it):
        ---
        {model_code[:4000]}
        ---

        Write a COMPLETE FastAPI application in a single file called `app.py`
        that does the following:

        1. Import FastAPI, torch, torchvision.transforms, PIL, io, base64.
        2. Re-define the EXACT same model class used in training.
        3. On startup, load the weights from "best_model.pt" onto CPU.
        4. Expose a POST endpoint at "/predict" that:
           a. Accepts a JSON body with a "image_base64" field (base64-encoded PNG).
           b. Decodes the image, applies the same transforms used in training.
           c. Runs inference and returns the predicted class index and
              confidence scores as JSON.
        5. Expose a GET endpoint at "/health" that returns {{"status": "ok"}}.
        6. Add `if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)`.

        Output ONLY the Python code. No markdown fences, no commentary.
    """)

    response = llm.invoke([
        SystemMessage(content="You are a senior MLOps engineer. Output ONLY valid Python code."),
        HumanMessage(content=prompt),
    ])

    deployment_code = response.content.strip()
    if deployment_code.startswith("```"):
        lines = deployment_code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        deployment_code = "\n".join(lines)

    # Write the deployment file to disk
    with open("app.py", "w") as f:
        f.write(deployment_code)

    print(f"[MLOpsEngineer] ✓ Generated app.py ({len(deployment_code)} chars)")

    return {
        "deployment_package": deployment_code,
    }


# ── Node 6: HF Publisher ─────────────────────────────────────────────────────

def hf_publisher(state: GraphState) -> dict:
    """
    Create a Hugging Face Hub repository and upload all artifacts.

    Uploads:
      - best_model.pt   (trained weights)
      - train.py        (training script)
      - app.py          (FastAPI server)
      - README.md       (auto-generated model card)

    Requires HF_TOKEN environment variable with write permissions.
    """
    from huggingface_hub import HfApi, create_repo

    llm = _get_llm(temperature=0.2)
    user_request = state["user_request"]
    model_code = state["model_code"]
    deployment_package = state["deployment_package"]

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[HFPublisher] ⚠ HF_TOKEN not set — skipping upload.")
        return {"hf_repo_url": "SKIPPED — no HF_TOKEN"}

    api = HfApi(token=hf_token)

    # Determine the username from the token
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"[HFPublisher] ✗ Failed to authenticate with HF: {e}")
        return {"hf_repo_url": f"ERROR — {e}"}

    # Generate a repo name from the user request using the LLM
    naming_prompt = textwrap.dedent(f"""\
        Given this ML task description, generate a short, URL-safe repository
        name for Hugging Face Hub (lowercase, hyphens only, max 40 chars).

        Task: {user_request}

        Output ONLY the repo name, nothing else. Example: "synthetic-image-classifier"
    """)

    naming_response = llm.invoke([
        SystemMessage(content="Output only a valid HF repo name. No quotes, no explanation."),
        HumanMessage(content=naming_prompt),
    ])
    repo_name = naming_response.content.strip().strip('"').strip("'")[:40]
    # Sanitize: keep only alphanumeric and hyphens
    repo_name = "".join(c if c.isalnum() or c == "-" else "-" for c in repo_name)
    repo_name = repo_name.strip("-") or "mas-trained-model"

    repo_id = f"{username}/{repo_name}"

    # Create the repo (or reuse if it exists)
    try:
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
        )
        print(f"[HFPublisher] ✓ Repo created/found: {repo_id}")
    except Exception as e:
        print(f"[HFPublisher] ✗ Failed to create repo: {e}")
        return {"hf_repo_url": f"ERROR — {e}"}

    # Generate a model card (README.md) using the LLM
    card_prompt = textwrap.dedent(f"""\
        Write a Hugging Face model card (README.md) in markdown for this model.

        Task: {user_request}
        Framework: PyTorch
        Architecture: Simple CNN (Conv2d → ReLU → MaxPool → Linear)
        Input: 3-channel 32×32 images
        Classes: 3 (class_0, class_1, class_2)
        Training: 5 epochs, Adam optimizer, CrossEntropyLoss

        Include:
        1. YAML frontmatter with: language: en, license: mit, library_name: pytorch,
           tags: [image-classification, pytorch, cnn]
        2. Model description
        3. How to use (load with torch.load)
        4. Training details
        5. Limitations (synthetic data)

        Output ONLY the markdown. No code fences wrapping the whole thing.
    """)

    card_response = llm.invoke([
        SystemMessage(content="You are a technical writer. Output valid markdown."),
        HumanMessage(content=card_prompt),
    ])
    model_card = card_response.content.strip()

    # Write the model card locally
    with open("MODEL_README.md", "w") as f:
        f.write(model_card)

    # Upload all artifacts to the repo
    files_to_upload = [
        ("best_model.pt", "best_model.pt"),
        ("train.py", "train.py"),
        ("app.py", "app.py"),
        ("MODEL_README.md", "README.md"),
    ]

    uploaded = []
    for local_path, hf_path in files_to_upload:
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message=f"Upload {hf_path} via MAS pipeline",
                )
                uploaded.append(hf_path)
                print(f"[HFPublisher]   ↑ Uploaded {hf_path}")
            except Exception as e:
                print(f"[HFPublisher]   ✗ Failed to upload {hf_path}: {e}")
        else:
            print(f"[HFPublisher]   ⚠ Skipping {local_path} (not found)")

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"[HFPublisher] ✓ Done! {len(uploaded)} files uploaded → {repo_url}")

    return {
        "hf_repo_url": repo_url,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Routing Logic (The Antigravity Engine)
# ──────────────────────────────────────────────────────────────────────────────

def route_after_critic(state: GraphState) -> str:
    """
    Conditional edge after the Critic node.

    Returns the name of the next node:
      - "systems_architect" → retry (error detected, budget remaining)
      - "mlops_engineer"    → success path
      - END                 → max retries exhausted
    """
    verdict = state.get("execution_logs", "")
    iteration_count = state.get("iteration_count", 0)

    if "SUCCESS" in verdict.upper():
        return "mlops_engineer"
    elif iteration_count >= 3:
        print("[Router] ⛔ Max iterations (3) reached. Stopping to prevent token burn.")
        return END
    else:
        print(f"[Router] 🔄 Routing back to systems_architect for retry …")
        return "systems_architect"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Graph Compilation
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assemble and compile the multi-agent LangGraph state machine."""

    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("data_curator", data_curator)
    workflow.add_node("systems_architect", systems_architect)
    workflow.add_node("executor", executor)
    workflow.add_node("critic", critic)
    workflow.add_node("mlops_engineer", mlops_engineer)
    workflow.add_node("hf_publisher", hf_publisher)

    # Set entry point
    workflow.set_entry_point("data_curator")

    # Linear edges
    workflow.add_edge("data_curator", "systems_architect")
    workflow.add_edge("systems_architect", "executor")
    workflow.add_edge("executor", "critic")

    # Conditional edge after critic (the retry / success / abort router)
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "systems_architect": "systems_architect",
            "mlops_engineer": "mlops_engineer",
            END: END,
        },
    )

    # MLOps → HF Publisher → END
    workflow.add_edge("mlops_engineer", "hf_publisher")
    workflow.add_edge("hf_publisher", END)

    return workflow.compile()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def run(user_request: str) -> GraphState:
    """
    Execute the full multi-agent pipeline.

    Args:
        user_request: Natural-language description of the ML task.

    Returns:
        The final GraphState dict after the pipeline completes.
    """
    graph = build_graph()

    initial_state: GraphState = {
        "user_request": user_request,
        "dataset_path": "",
        "model_code": "",
        "execution_logs": "",
        "deployment_package": "",
        "iteration_count": 0,
        "hf_repo_url": "",
    }

    print("=" * 72)
    print("  AUTONOMOUS ML AGENT SYSTEM — STARTING PIPELINE")
    print("=" * 72)
    print(f"  Request: {user_request}")
    print("=" * 72)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)

    if final_state.get("deployment_package"):
        print("  ✓ Model trained and deployment server generated.")
        print("  ✓ Files created:")
        print("    • synthetic_dataset/   (training data)")
        print("    • train.py             (training script)")
        print("    • best_model.pt        (saved weights)")
        print("    • app.py               (FastAPI server)")
        hf_url = final_state.get("hf_repo_url", "")
        if hf_url and "ERROR" not in hf_url and "SKIPPED" not in hf_url:
            print(f"\n  ✓ Model published to Hugging Face:")
            print(f"    🤗 {hf_url}")
        elif "SKIPPED" in hf_url:
            print(f"\n  ⚠ HF upload skipped (set HF_TOKEN to enable).")
        else:
            print(f"\n  ✗ HF upload failed: {hf_url}")
        print(f"\n  To start the server locally:\n    $ python app.py")
    else:
        print("  ✗ Pipeline ended without a deployment package.")
        print(f"  Iterations used: {final_state.get('iteration_count', '?')}/3")
        print(f"  Last logs: {final_state.get('execution_logs', 'N/A')[:300]}")

    print("=" * 72)
    return final_state


# ──────────────────────────────────────────────────────────────────────────────
# 8. Execute (for Colab / direct run)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    final = run(
        user_request=(
            "Build an image classifier that can distinguish between "
            "three categories of synthetic patterns."
        )
    )
