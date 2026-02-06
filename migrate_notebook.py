import json
import os

input_file = "[TEMPLATE][v0_17][AGNOSTIC_AGENT][Stack__QWEN].ipynb"
output_file = "[TEMPLATE][v0_18][AGNOSTIC_AGENT][Stack__QWEN].ipynb"

# ---------------------------------------------------------
# CELL CONTENT DEFINITIONS
# ---------------------------------------------------------

# 1. INSTALL CELL (V7 - Force Update)
new_install_source = [
    "# ============================================================\n",
    "# @title INSTALL – Agnostic Agent (GitHub) + Infra Qwen3\n",
    "# ============================================================\n",
    "import os, sys, subprocess, importlib\n",
    "\n",
    "def sh(cmd: str):\n",
    "    return subprocess.run(cmd, shell=True, text=True, capture_output=True)\n",
    "\n",
    "def install_if_missing(package_name, import_name=None):\n",
    "    import_name = import_name or package_name.split('>=')[0].split('==')[0].replace('-', '_')\n",
    "    try: \n",
    "        importlib.import_module(import_name)\n",
    "    except ImportError:\n",
    "        print(f\"📦 Installing {package_name}...\")\n",
    "        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", package_name])\n",
    "\n",
    "# 1. SUPER CLEANUP & UPDATE\n",
    "if os.path.exists(\"repo_agnostic\"):\n",
    "    print(\"🔄 Repo exists. Pulling latest changes...\")\n",
    "    os.system(\"cd repo_agnostic && git pull\")\n",
    "else:\n",
    "    print(\"🧹 Cleaning up shadowing folders...\")\n",
    "    os.system(\"rm -rf agnostic_agent repo_agnostic\")\n",
    "    print(\"📥 Cloning repo...\")\n",
    "    os.system(\"git clone https://github.com/JacoboGGLeon/agnostic_agent.git repo_agnostic\")\n",
    "\n",
    "# Always reinstall editable to be sure (fast/idempotent usually)\n",
    "print(\"🛠 Updating package install...\")\n",
    "os.system(\"pip install -q -e repo_agnostic\")\n",
    "\n",
    "# 2. FIX PATH\n",
    "pkg_path = os.path.abspath(\"repo_agnostic\")\n",
    "if pkg_path not in sys.path: sys.path.insert(0, pkg_path)\n",
    "\n",
    "# 3. OTHER DEPENDENCIES (Idempotent)\n",
    "deps = [\n",
    "    (\"vllm>=0.9.0\", \"vllm\"),\n",
    "    (\"huggingface_hub>=0.23.0\", \"huggingface_hub\"),\n",
    "    (\"openai==1.108.2\", \"openai\"),\n",
    "    (\"langchain\", \"langchain\"),\n",
    "    (\"langchain-core\", \"langchain_core\"),\n",
    "    (\"langchain-openai\", \"langchain_openai\"),\n",
    "    (\"langgraph\", \"langgraph\"),\n",
    "    (\"pydantic>=2.7.0\", \"pydantic\"),\n",
    "    (\"pyyaml>=6.0\", \"yaml\"),\n",
    "    (\"streamlit\", \"streamlit\"),\n",
    "    (\"sqlite-vec\", \"sqlite_vec\"),\n",
    "    (\"pandas\", \"pandas\"),\n",
    "    (\"numpy\", \"numpy\"),\n",
    "]\n",
    "\n",
    "for pkg, imp in deps:\n",
    "    install_if_missing(pkg, imp)\n",
    "\n",
    "try:\n",
    "    importlib.import_module(\"langchain_qwq_vllm\")\n",
    "except ImportError:\n",
    "    print(\"📦 Installing langchain-qwq-vllm...\")\n",
    "    os.system(\"pip install -q git+https://github.com/whynpc9/langchain-qwq-vllm.git\")\n",
    "\n",
    "print(\"✅ Agnostic Agent INSTALLED and ready!\")\n"
]

# 2. STREAMLIT APP COPY
new_app_source = [
    "#@title streamlit_app.py (From Git)\n",
    "import os, shutil\n",
    "\n",
    "if os.path.exists(\"repo_agnostic/streamlit_app.py\"):\n",
    "    shutil.copy(\"repo_agnostic/streamlit_app.py\", \".\")\n",
    "    print(\"✅ streamlit_app.py copied from repo_agnostic\")\n",
    "else:\n",
    "    print(\"❌ ERROR: streamlit_app.py not found in repo_agnostic\")\n"
]

# 3. STREAMLIT CONFIG COPY
new_config_source = [
    "#@title .streamlit/config.toml (From Git)\n",
    "import os, shutil\n",
    "\n",
    "!mkdir -p .streamlit\n",
    "if os.path.exists(\"repo_agnostic/.streamlit/config.toml\"):\n",
    "    shutil.copy(\"repo_agnostic/.streamlit/config.toml\", \".streamlit/config.toml\")\n",
    "    print(\"✅ .streamlit/config.toml copied from repo_agnostic\")\n",
    "else:\n",
    "    print(\"❌ ERROR: .streamlit/config.toml not found in repo_agnostic\")\n"
]


# ---------------------------------------------------------
# MIGRATION LOGIC
# ---------------------------------------------------------

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found locally.")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = []
for cell in nb.get("cells", []):
    source = cell.get("source", [])
    source_text = "".join(source)
    
    # INSTALL CELL
    if "@title INSTALL" in source_text:
        print("Build: Injection of Cleanup + Update + Install logic...")
        cell["source"] = new_install_source
        new_cells.append(cell)
        continue

    # REMOVE SHADOWING CELLS (mkdir or writefile agnostic_agent/)
    if "%%writefile agnostic_agent/" in source_text or "!mkdir -p agnostic_agent" in source_text:
        # print(f"Build: Removing shadowing cell ({len(source_text)} chars)")
        continue

    # STREAMLIT APP -> COPY FROM GIT
    if "#@title streamlit_app.py" in source_text or "%%writefile streamlit_app.py" in source_text:
        print("Build: Replacing streamlit_app.py writefile with Git Copy logic...")
        cell["source"] = new_app_source
        cell["outputs"] = []
        new_cells.append(cell)
        continue

    # CONFIG TOML -> COPY FROM GIT
    if "#@title .streamlit/config.toml" in source_text or "%%writefile .streamlit/config.toml" in source_text:
        print("Build: Replacing .streamlit/config.toml writefile with Git Copy logic...")
        cell["source"] = new_config_source
        cell["outputs"] = []
        new_cells.append(cell)
        continue
    
    # Keep other cells
    new_cells.append(cell)

nb["cells"] = new_cells

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Build: Success! Generated {output_file}")
