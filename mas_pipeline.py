"""
Autonomous Multi-Agent System (MAS) for Neural Network Curation, Training & Deployment
========================================================================================

A LangGraph state machine that orchestrates 6 specialized agents:
  1. Data Curator      – Generates synthetic datasets
  2. Systems Architect – Writes PyTorch training scripts via Groq LLM
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
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

# ──────────────────────────────────────────────────────────────────────────────
# 2. State Definition
# ──────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state that flows through every node in the graph."""
    user_request: str        # The original user prompt describing the task
    dataset_path: str        # Filesystem path to the generated dataset
    class_names: str         # Comma-separated list of discovered class names
    model_code: str          # Full Python source of the training script
    execution_logs: str      # Combined stdout + stderr from training
    deployment_package: str  # Full Python source of the Gradio app
    iteration_count: int     # Number of architect→executor→critic cycles
    hf_repo_url: str         # Hugging Face repo URL after upload


# ──────────────────────────────────────────────────────────────────────────────
# 3. LLM Factory (with retry for rate limits)
# ──────────────────────────────────────────────────────────────────────────────

# ── Change this to switch models globally ──
# Groq-supported models (all free tier with generous limits):
#   "llama-3.3-70b-versatile"    ← best quality, 70B params
#   "llama-3.1-8b-instant"       ← fastest, 8B params
#   "mixtral-8x7b-32768"         ← Mixtral MoE, 32k context
#   "gemma2-9b-it"               ← Google Gemma 2, 9B params
MODEL_NAME = "llama-3.3-70b-versatile"

import time


def _invoke_with_retry(llm, messages, max_retries: int = 5):
    """Invoke the LLM with exponential backoff on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = min(2 ** attempt * 10, 120)  # 10s, 20s, 40s, 80s, 120s
                print(f"[RateLimit] ⏳ Quota hit — waiting {wait}s before retry {attempt + 1}/{max_retries} …")
                time.sleep(wait)
            else:
                raise  # Non-rate-limit errors propagate immediately
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")


def _get_llm(temperature: float = 0.2) -> ChatGroq:
    """Return a ChatGroq text LLM instance."""
    return ChatGroq(
        model=MODEL_NAME,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=temperature,
        max_tokens=8192,
    )


# ── Vision model for MLLM labeling ──
VISION_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_vision_llm(temperature: float = 0.0) -> ChatGroq:
    """Return a ChatGroq vision-capable LLM for image classification."""
    return ChatGroq(
        model=VISION_MODEL_NAME,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=temperature,
        max_tokens=256,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Agent Nodes
# ──────────────────────────────────────────────────────────────────────────────

# ── Node 1: Data Curator (Real Images + MLLM Labeling) ───────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def data_curator(state: GraphState) -> dict:
    """
    Download real-world images and label them using a multimodal LLM.

    Stage 1: Use the text LLM to write a script that downloads real images
             from HuggingFace datasets (based on the user's request).
    Stage 2: Use a Groq vision model (MLLM) to classify each image into
             categories derived from the user request.
    """
    import base64
    import glob
    import json
    from pathlib import Path

    llm = _get_llm()
    user_request = state["user_request"]

    # ── Stage 1: Determine categories and download images ────────────────
    print("[DataCurator] Stage 1: Determining categories and downloading images …")

    planning_prompt = textwrap.dedent(f"""\
        You are a data-engineering agent. The user wants to build a model for:

        >>> {user_request} <<< 

        Step 1: Identify 3 to 5 class categories that make sense for this task.
        Step 2: Decide how many images per class to download. Use your judgment
                based on the complexity of the task:
                - Simple tasks (e.g., binary classification, few classes): 30-50 per class
                - Moderate tasks (e.g., 3-5 classes, some visual similarity): 50-100 per class
                - Complex tasks (e.g., fine-grained categories, subtle differences): 100-200 per class
                Print your decision as: "VOLUME: <N> images per class"
        Step 3: Write a COMPLETE Python script that:
          - Uses the `datasets` library from HuggingFace to download a relevant
            real-world image dataset. Pick one that closely matches the task.
            Good options: "cifar10", "food101", "cats_vs_dogs", "fashion_mnist"
          - Downloads images based on your volume decision from Step 2.
          - You MUST create the directory: `os.makedirs("./real_dataset/raw/", exist_ok=True)`
          - Saves images as PNG files into "./real_dataset/raw/" folder.
          - Each image should be saved as "img_NNNN.png" (sequential numbering).
          - Print the class names as: "CLASSES: cat, dog, bird"
          - Print the total count as: "TOTAL: <N> images downloaded"
          - Print "Download complete" when done.
          - The script must handle the case where images are PIL Image objects
            OR numpy arrays.
          - Use `split="train"` and slice with `select(range(N))` to limit.

        Output ONLY the Python code. No markdown fences, no commentary.
    """)

    for attempt in range(3):
        response = _invoke_with_retry(llm, [
            SystemMessage(content="You are a senior data engineer. Output ONLY executable Python code."),
            HumanMessage(content=planning_prompt),
        ])

        download_script = _strip_code_fences(response.content)

        script_path = f"download_images_{attempt}.py"
        with open(script_path, "w") as f:
            f.write(download_script)

        print(f"[DataCurator]   Running download script (attempt {attempt+1}/3) …")
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        stdout = result.stdout
        if result.returncode == 0:
            print(f"[DataCurator] ✓ {stdout.strip()[-200:]}")
            break
        else:
            print(f"[DataCurator] ⚠ Download script error:\n{result.stderr[:500]}")
            planning_prompt += textwrap.dedent(f"""\
            
            ⚠ PREVIOUS ATTEMPT FAILED WITH ERROR:
            {result.stderr[-1000:]}
            
            You MUST fix the error. If the dataset cannot be loaded (e.g., config name needed, trust_remote_code=True, or dataset doesn't exist), pick a DIFFERENT fallback dataset from HuggingFace (e.g., 'cifar10', 'cats_vs_dogs', 'food101') instead.
            """)
    else:
        print("[DataCurator] ⚠ All download attempts failed. The dataset will be empty.")

    # Extract class names from script output
    class_names = []
    for line in stdout.split("\n"):
        if line.strip().upper().startswith("CLASSES:"):
            raw = line.split(":", 1)[1].strip()
            class_names = [c.strip() for c in raw.split(",") if c.strip()]
            break

    if not class_names:
        # Fallback: ask LLM for class names
        fallback = _invoke_with_retry(llm, [
            SystemMessage(content="Output ONLY a comma-separated list of class names. Nothing else."),
            HumanMessage(content=f"List 3-5 image classification categories for: {user_request}"),
        ])
        class_names = [c.strip() for c in fallback.content.strip().split(",") if c.strip()]

    if not class_names:
        class_names = ["class_0", "class_1", "class_2"]

    print(f"[DataCurator]   Classes: {class_names}")

    # ── Stage 2: MLLM labeling with vision model ─────────────────────────
    print("[DataCurator] Stage 2: Labeling images with vision MLLM …")

    raw_dir = Path("./real_dataset/raw")
    labeled_dir = Path("./real_dataset/labeled")

    # Create class subdirectories
    for cls in class_names:
        (labeled_dir / cls).mkdir(parents=True, exist_ok=True)

    # Gather all images
    image_files = sorted(
        glob.glob(str(raw_dir / "*.png"))
        + glob.glob(str(raw_dir / "*.jpg"))
        + glob.glob(str(raw_dir / "*.jpeg"))
    )

    if not image_files:
        print("[DataCurator] ⚠ No images found in raw dir, falling back to unlabeled.")
        return {
            "dataset_path": str(raw_dir),
            "class_names": ",".join(class_names),
        }

    vision_llm = _get_vision_llm()
    classes_str = ", ".join(class_names)
    labeled_count = {cls: 0 for cls in class_names}
    total = len(image_files)  # Label all downloaded images

    from langchain_core.messages import HumanMessage as HM

    for i, img_path in enumerate(image_files[:total]):
        try:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Determine MIME type
            ext = Path(img_path).suffix.lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
                ext.lstrip("."), "image/png"
            )

            label_msg = HM(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Classify this image into EXACTLY ONE of these categories: {classes_str}\n"
                            f"Respond with ONLY the category name, nothing else."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                    },
                ]
            )

            label_response = _invoke_with_retry(vision_llm, [label_msg], max_retries=3)
            predicted_label = label_response.content.strip().lower()

            # Match to closest class name
            matched_class = None
            for cls in class_names:
                if cls.lower() in predicted_label or predicted_label in cls.lower():
                    matched_class = cls
                    break
            if not matched_class:
                matched_class = class_names[0]  # Default fallback

            # Copy image to labeled directory
            import shutil
            dest = labeled_dir / matched_class / Path(img_path).name
            shutil.copy2(img_path, dest)
            labeled_count[matched_class] += 1

            if (i + 1) % 10 == 0:
                print(f"[DataCurator]   Labeled {i + 1}/{total} images …")

        except Exception as e:
            print(f"[DataCurator]   ⚠ Failed to label {Path(img_path).name}: {e}")
            continue

    print(f"[DataCurator] ✓ Labeling complete: {dict(labeled_count)}")

    return {
        "dataset_path": str(labeled_dir),
        "class_names": ",".join(class_names),
    }


# ── Node 2: Systems Architect ────────────────────────────────────────────────

def systems_architect(state: GraphState) -> dict:
    """
    Use the LLM to write a complete PyTorch training script.

    If previous execution logs contain errors (from a retry loop), they are
    included in the prompt so the LLM can fix the issues.
    """
    llm = _get_llm(temperature=0.3)
    user_request = state["user_request"]
    dataset_path = state["dataset_path"]
    class_names = state.get("class_names", "")
    num_classes = max(len(class_names.split(",")), 2) if class_names else 3
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
        CLASS NAMES: {class_names}
        NUMBER OF CLASSES: {num_classes}
        (The dataset has subdirectories named after each class with real-world PNG/JPG images.)

        {error_context}

        Write a COMPLETE, SELF-CONTAINED Python training script called `train.py`.

        Hard requirements:
        1. Use PyTorch and torchvision.
        2. Load the dataset from "{dataset_path}" using
           torchvision.datasets.ImageFolder and a DataLoader.
        3. Define a simple CNN (Conv2d → ReLU → MaxPool → Flatten → Linear).
           The CNN must accept 3-channel 64×64 images and output {num_classes} classes.
        4. Use CrossEntropyLoss and Adam optimizer (lr=1e-3).
        5. Train for exactly 5 epochs.
        6. After each epoch, print: "Epoch {{epoch}}/5 — Loss: {{avg_loss:.4f}}"
        7. After training, save the model weights to "best_model.pt" using
           torch.save(model.state_dict(), "best_model.pt").
        8. Print "Training complete. Model saved to best_model.pt" at the end.
        9. Handle the case where CUDA is not available by falling back to CPU.
        10. Use torchvision.transforms to Resize(64), ToTensor(), and Normalize.
        11. Use num_workers=0 in DataLoader (required for Kaggle).

        Output ONLY the Python code. No markdown fences, no commentary.
    """)

    response = _invoke_with_retry(llm, [
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

    response = _invoke_with_retry(llm, [
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


# ── Node 5: MLOps Engineer (Gradio + Model Download) ─────────────────────────

def mlops_engineer(state: GraphState) -> dict:
    """
    Generate a Gradio app that serves the trained model with a download button.
    """
    llm = _get_llm(temperature=0.2)
    user_request = state["user_request"]
    model_code = state["model_code"]
    class_names = state.get("class_names", "class_0,class_1,class_2")

    prompt = textwrap.dedent(f"""\
        You are a senior MLOps engineer.

        The user trained a PyTorch model for: {user_request}
        The class names are: {class_names}

        Here is the training script that was used (you need the model class
        definition from it):
        ---
        {model_code[:4000]}
        ---

        Write a COMPLETE Gradio application in a single file called `app.py`
        that does the following:

        1. Import gradio, torch, torchvision.transforms, PIL.
        2. Re-define the EXACT same model class used in training.
        3. Load the weights from "best_model.pt" onto CPU at module level.
        4. Define the class names as a list: {class_names.split(",")}.
        5. Create a prediction function that:
           a. Takes a PIL Image as input.
           b. Applies the same transforms used in training (Resize(64),
              ToTensor(), Normalize).
           c. Runs inference and returns a dict mapping class names to
              confidence scores (use torch.nn.functional.softmax).
        6. Build a Gradio Interface with:
           a. gr.Image(type="pil") as input.
           b. gr.Label(num_top_classes=len(class_names)) as output.
           c. A title like "Image Classifier — {{task description}}".
           d. A description explaining what the model does.
        7. IMPORTANT: Add a model download component. Use gr.File with
           value="best_model.pt" so users can download the trained weights.
           Place it below the main interface using gr.Blocks layout:

           with gr.Blocks() as demo:
               gr.Markdown("# Title")
               with gr.Row():
                   image_input = gr.Image(type="pil")
                   label_output = gr.Label()
               predict_btn = gr.Button("Classify")
               predict_btn.click(predict, inputs=image_input, outputs=label_output)
               gr.Markdown("### Download Model")
               gr.File(value="best_model.pt", label="Download trained model weights")

        8. Launch with: demo.launch()

        Output ONLY the Python code. No markdown fences, no commentary.
    """)

    response = _invoke_with_retry(llm, [
        SystemMessage(content="You are a senior MLOps engineer. Output ONLY valid Python code."),
        HumanMessage(content=prompt),
    ])

    deployment_code = _strip_code_fences(response.content)

    # Write the deployment file to disk
    with open("app.py", "w") as f:
        f.write(deployment_code)

    print(f"[MLOpsEngineer] ✓ Generated Gradio app.py ({len(deployment_code)} chars)")

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

    naming_response = _invoke_with_retry(llm, [
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

    card_response = _invoke_with_retry(llm, [
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
        "class_names": "",
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
        class_names = final_state.get("class_names", "")
        print("  ✓ Model trained and Gradio app generated.")
        print(f"  ✓ Classes: {class_names}")
        print("  ✓ Files created:")
        print("    • real_dataset/        (AI-labeled training data)")
        print("    • train.py             (training script)")
        print("    • best_model.pt        (saved weights)")
        print("    • app.py               (Gradio app + model download)")
        hf_url = final_state.get("hf_repo_url", "")
        if hf_url and "ERROR" not in hf_url and "SKIPPED" not in hf_url:
            print(f"\n  ✓ Model published to Hugging Face:")
            print(f"    🤗 {hf_url}")
        elif "SKIPPED" in hf_url:
            print(f"\n  ⚠ HF upload skipped (set HF_TOKEN to enable).")
        else:
            print(f"\n  ✗ HF upload failed: {hf_url}")
        print(f"\n  To launch Gradio:\n    $ python app.py")
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
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous ML Agent System")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Build an image classifier that can distinguish between cats, dogs, and birds using real-world photos.",
        help="The natural language request for the model to build."
    )
    args = parser.parse_args()
    
    final = run(user_request=args.prompt)
