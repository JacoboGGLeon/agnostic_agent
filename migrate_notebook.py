import json
import os

input_file = "[TEMPLATE][v0_17][AGNOSTIC_AGENT][Stack__QWEN].ipynb"
output_file = "[TEMPLATE][v0_18][AGNOSTIC_AGENT][Stack__QWEN].ipynb"

# New installation cell content (v4 - The "Purge" Edition)
new_install_source = [
    "# ============================================================\n",
    "# @title INSTALL – Agnostic Agent (GitHub) + Infra Qwen3\n",
    "# ============================================================\n",
    "# 1. SUPER CLEANUP (Delete conflicting folders)\n",
    "!rm -rf agnostic_agent  # <--- Deletes the lingering folder that causes ImportError\n",
    "!rm -rf repo_agnostic   # <--- Clean start\n",
    "\n",
    "# 2. CLONE REPO (Into 'repo_agnostic' to avoid any persistent name clash)\n",
    "!git clone https://github.com/JacoboGGLeon/agnostic_agent.git repo_agnostic\n",
    "!pip install -e repo_agnostic\n",
    "\n",
    "# 3. FIX PATH (Tell Python where the package is)\n",
    "import sys, os\n",
    "pkg_path = os.path.abspath(\"repo_agnostic\")\n",
    "if pkg_path not in sys.path: sys.path.insert(0, pkg_path)\n",
    "\n",
    "# 4. OTHER DEPENDENCIES\n",
    "!pip -q install \"vllm>=0.9.0\" \"huggingface_hub>=0.23.0\" \"openai==1.108.2\"\n",
    "!pip -q install langchain langchain-core langchain-openai langgraph\n",
    "!pip -q install \"git+https://github.com/whynpc9/langchain-qwq-vllm.git\"\n",
    "!pip -q install \"pydantic>=2.7.0\" \"pyyaml>=6.0\" streamlit sqlite-vec pandas numpy\n",
    "\n",
    "print(\"✅ Agnostic Agent INSTALLED from GitHub!\")\n",
    "print(\"📂 Repo Path: repo_agnostic\")\n",
    "print(\"⚠️ IMPORTANT: If imports fail, go to Runtime -> Restart session, then run DOWNLOAD MODELS directly.\")\n"
]

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found locally. Please ensure the v0.17 template is present.")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = []
for cell in nb.get("cells", []):
    source = cell.get("source", [])
    source_text = "".join(source)
    
    # 1. Update Install Cell
    if "@title INSTALL" in source_text:
        print("Build: Injection of Cleanup + Install logic...")
        cell["source"] = new_install_source
        new_cells.append(cell)
        continue

    # 2. Skip shadowing cells (removed 12+ cells in total)
    if "%%writefile agnostic_agent/" in source_text or "!mkdir -p agnostic_agent" in source_text:
        # Check if this cell does ONLY mkdir, or if it writes a file to that dir
        print(f"Build: Removed shadowing cell -> {source[0].strip()[:50]}...")
        continue

    new_cells.append(cell)

nb["cells"] = new_cells

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Build: Success! Generated {output_file}")
