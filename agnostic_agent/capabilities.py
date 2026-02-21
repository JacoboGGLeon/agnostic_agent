from __future__ import annotations

"""
Infraestructura de *capacidades* (modelos/servidores) para el Agnostic Deep Agent 2026.

Incluye:
- Descarga de modelos Qwen3 (LLM / Embeddings / Reranker) desde Hugging Face.
- Lanzamiento de servidores vLLM OpenAI-compatible (LLM / EMB / RERANK).
- Configuración del planner (PlannerConfig) y construcción del LLM planner
#   (build_planner_llm + build_planner_system_message) sobre ChatOpenAI (generic).

NOTA:
- Esta lógica es agnóstica del dominio; sólo gestiona modelos y servidores.
- El wiring con LangGraph y el resto del agente se hace en logic.py y agent.py.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Any
import os
import sys
import subprocess
import time
import socket
import json
import urllib.request
import pathlib

from huggingface_hub import snapshot_download
from langchain_core.messages import SystemMessage
try:
    from langchain_qwq import ChatQwenVllm as ChatGenVllm
    HAS_LANGCHAIN_QWQ = True
except Exception:
    from langchain_openai import ChatOpenAI as ChatGenVllm
    HAS_LANGCHAIN_QWQ = False


# ─────────────────────────────────────────────
# Helpers básicos
# ─────────────────────────────────────────────

def _str_to_bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "on")


# ─────────────────────────────────────────────
# Descarga de modelos (HF snapshot)
# ─────────────────────────────────────────────

DEFAULT_LLM_ID = "Qwen/Qwen3-0.6B" # Example default
DEFAULT_EMB_ID = "Qwen/Qwen3-Embedding-0.6B" # Example default
DEFAULT_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B" # Example default

@dataclass
class LocalModelPaths:
    """Rutas locales a los modelos usados por el agente."""
    llm_dir: str
    emb_dir: str
    rerank_dir: str


def ensure_model_downloaded(model_id: str, local_dir: pathlib.Path) -> str:
    """
    Descarga el modelo de Hugging Face a `local_dir` si no existe.
    Devuelve la ruta absoluta como string.
    """
    local_dir = pathlib.Path(local_dir)
    if local_dir.is_dir():
        return str(local_dir.resolve())

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.png", "*.jpg", "LICENSE*"],
    )
    return str(local_dir.resolve())


def prepare_local_models(
    llm_model_id: Optional[str] = None,
    emb_model_id: Optional[str] = None,
    rerank_model_id: Optional[str] = None,
    base_dir: str | os.PathLike = "LM_MODEL",
) -> LocalModelPaths:
    """
    Descarga (si es necesario) los modelos y devuelve sus rutas locales.

    Si algún ID no se pasa, se lee de las variables de entorno o se usan defaults:
    - LLM_MODEL_ID
    - EMB_MODEL_ID
    - RERANK_MODEL_ID
    """
    base_dir_path = pathlib.Path(base_dir)
    base_dir_path.mkdir(parents=True, exist_ok=True)

    llm_model_id = llm_model_id or os.getenv("LLM_MODEL_ID", DEFAULT_LLM_ID)
    emb_model_id = emb_model_id or os.getenv("EMB_MODEL_ID", DEFAULT_EMB_ID)
    rerank_model_id = rerank_model_id or os.getenv("RERANK_MODEL_ID", DEFAULT_RERANK_ID)

    llm_dir = ensure_model_downloaded(llm_model_id, base_dir_path / "LLM_MAIN")
    emb_dir = ensure_model_downloaded(emb_model_id, base_dir_path / "Embedding")
    rerank_dir = ensure_model_downloaded(rerank_model_id, base_dir_path / "Reranker")

    return LocalModelPaths(llm_dir=llm_dir, emb_dir=emb_dir, rerank_dir=rerank_dir)


# ─────────────────────────────────────────────
# Servidores vLLM (OpenAI-compatible)
# ─────────────────────────────────────────────

@dataclass
class VllmConfig:
    # Host / puertos
    host: str = "127.0.0.1"
    llm_port: int = int(os.getenv("LLM_PORT", "8000"))
    emb_port: int = int(os.getenv("EMB_PORT", "8001"))
    rerank_port: int = int(os.getenv("RERANK_PORT", "8002"))

    # Presupuesto de VRAM por servidor
    llm_gpu_util: float = float(os.getenv("VLLM_LLM_GPU_UTIL", "0.4"))
    emb_gpu_util: float = float(os.getenv("VLLM_EMB_GPU_UTIL", "0.3"))
    rerank_gpu_util: float = float(os.getenv("VLLM_RERANK_GPU_UTIL", "0.3"))

    # Longitud máxima de contexto por servidor
    llm_max_len: int = int(os.getenv("VLLM_LLM_MAX_LEN", "2048"))
    emb_max_len: int = int(os.getenv("VLLM_EMB_MAX_LEN", "1024"))
    rerank_max_len: int = int(os.getenv("VLLM_RERANK_MAX_LEN", "1024"))

    # Concurrencia (nº de secuencias simultáneas)
    llm_max_num_seqs: int = int(os.getenv("VLLM_LLM_MAX_NUM_SEQS", "4"))
    emb_max_num_seqs: int = int(os.getenv("VLLM_EMB_MAX_NUM_SEQS", "4"))
    rerank_max_num_seqs: int = int(os.getenv("VLLM_RERANK_MAX_NUM_SEQS", "4"))

    # Nombres "served_model_name" dentro de vLLM
    llm_served_name: str = os.getenv("LLM_SERVED_NAME", "agnostic-llm")
    emb_served_name: str = os.getenv("EMB_SERVED_NAME", "agnostic-embed")
    rerank_served_name: str = os.getenv("RERANK_SERVED_NAME", "agnostic-rerank")

    # Parser de tool-calls y razonamiento
    # Default: "qwen3_xml" (o "tool_calling_system" según soporte de vLLM)
    tool_call_parser: str = os.getenv("VLLM_TOOL_CALL_PARSER", "qwen3_xml")
    enable_reasoning: bool = _str_to_bool(os.getenv("VLLM_ENABLE_REASONING", "1"))
    reasoning_parser: Optional[str] = os.getenv("VLLM_REASONING_PARSER", "qwen3")

    # Flags para arrancar (o no) servidores extra
    start_emb_server: bool = _str_to_bool(os.getenv("VLLM_START_EMB_SERVER", "0"))
    start_rerank_server: bool = _str_to_bool(os.getenv("VLLM_START_RERANK_SERVER", "0"))


@dataclass
class VllmServers:
    llm_proc: subprocess.Popen
    emb_proc: Optional[subprocess.Popen] = None
    rerank_proc: Optional[subprocess.Popen] = None
    llm_log_path: str = ""
    emb_log_path: Optional[str] = None
    rerank_log_path: Optional[str] = None


@dataclass
class VllmEndpoints:
    llm_base_url: str
    emb_base_url: Optional[str] = None
    rerank_base_url: Optional[str] = None


def _free_port_if_needed(host: str, port: int) -> None:
    try:
        s = socket.socket()
        s.settimeout(0.5)
        if s.connect_ex((host, port)) == 0:
            _ = os.system(f"fuser -k {port}/tcp || true")
        s.close()
    except Exception:
        pass


def _url_open_no_proxy(url: str, timeout: float = 2.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _wait_until_ready(
    url: str,
    server_proc: subprocess.Popen,
    log_path: str,
    seconds: int = 180,
    sleep: float = 2.0,
) -> bool:
    start = time.time()
    while time.time() - start < seconds:
        if server_proc.poll() is not None:
            print(f"❌ Servidor terminó con código {server_proc.returncode}. Log tail:")
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
                print("\n".join(lines[-80:]))
            except Exception:
                pass
            return False
        try:
            _ = _url_open_no_proxy(url, timeout=2.5)
            return True
        except Exception:
            time.sleep(sleep)
    print("⏰ Timeout esperando servidor, mostrando tail del log:")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
        print("\n".join(lines[-80:]))
    except Exception:
        pass
    return False


def _launch_vllm_server(
    name: str,
    model_dir: str,
    host: str,
    port: int,
    served_model_name: Optional[str],
    gpu_util: float,
    max_model_len: Optional[int],
    max_num_seqs: Optional[int],
    extra_flags: Optional[list[str]] = None,
) -> tuple[subprocess.Popen, str]:
    """
    Lanza un servidor vLLM OpenAI-compatible para un modelo dado.
    """
    _free_port_if_needed(host, port)
    log_path = f"vllm_{name}.log"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_util),
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if extra_flags:
        cmd += list(extra_flags)

    print(f"\n🚀 Lanzando servidor vLLM [{name}] en puerto {port}")
    print(" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    return proc, log_path


def start_local_vllm_servers(
    model_paths: LocalModelPaths,
    config: Optional[VllmConfig] = None,
    set_env: bool = True,
) -> tuple[VllmEndpoints, VllmServers]:
    """
    Lanza servidores vLLM:

      - LLM (generate, con tool-calling + reasoning de Qwen3)
      - (Opcional) Embeddings (embed)
      - (Opcional) Reranker (score)
    """
    cfg = config or VllmConfig()

    # LLM – flags alineados a vLLM + langchain driver
    llm_extra_flags: list[str] = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        cfg.tool_call_parser,
    ]
    if cfg.enable_reasoning and cfg.reasoning_parser:
        llm_extra_flags += ["--reasoning-parser", cfg.reasoning_parser]

    llm_proc, llm_log = _launch_vllm_server(
        name="language",
        model_dir=model_paths.llm_dir,
        host=cfg.host,
        port=cfg.llm_port,
        served_model_name=cfg.llm_served_name,
        gpu_util=cfg.llm_gpu_util,
        max_model_len=cfg.llm_max_len,
        max_num_seqs=cfg.llm_max_num_seqs,
        extra_flags=llm_extra_flags,
    )

    emb_proc = None
    emb_log = None
    if cfg.start_emb_server:
        emb_proc, emb_log = _launch_vllm_server(
            name="embedding",
            model_dir=model_paths.emb_dir,
            host=cfg.host,
            port=cfg.emb_port,
            served_model_name=cfg.emb_served_name,
            gpu_util=cfg.emb_gpu_util,
            max_model_len=cfg.emb_max_len,
            max_num_seqs=cfg.emb_max_num_seqs,
        )

    rerank_proc = None
    rerank_log = None
    if cfg.start_rerank_server:
        rerank_proc, rerank_log = _launch_vllm_server(
            name="reranker",
            model_dir=model_paths.rerank_dir,
            host=cfg.host,
            port=cfg.rerank_port,
            served_model_name=cfg.rerank_served_name,
            gpu_util=cfg.rerank_gpu_util,
            max_model_len=cfg.rerank_max_len,
            max_num_seqs=cfg.rerank_max_num_seqs,
        )

    llm_base = f"http://{cfg.host}:{cfg.llm_port}/v1"
    emb_base = f"http://{cfg.host}:{cfg.emb_port}/v1" if cfg.start_emb_server else None
    rerank_base = f"http://{cfg.host}:{cfg.rerank_port}/v1" if cfg.start_rerank_server else None

    print("\n⏳ Esperando LLM server...")
    ok_llm = _wait_until_ready(f"{llm_base}/models", llm_proc, llm_log)

    ok_emb = True
    if cfg.start_emb_server and emb_proc is not None and emb_log is not None:
        print("⏳ Esperando Embedding server...")
        ok_emb = _wait_until_ready(f"{emb_base}/models", emb_proc, emb_log)

    ok_rerank = True
    if cfg.start_rerank_server and rerank_proc is not None and rerank_log is not None:
        print("⏳ Esperando Reranker server...")
        ok_rerank = _wait_until_ready(f"{rerank_base}/models", rerank_proc, rerank_log)

    if not ok_llm or not ok_emb or not ok_rerank:
        raise SystemExit("❌ Algún servidor vLLM no quedó listo. Revisa logs.")

    print("\n✅ Servidores vLLM listos.")
    print("\n📋 Modelos en LLM server:")
    print(_url_open_no_proxy(f"{llm_base}/models"))
    if emb_base is not None:
        print("\n📋 Modelos en Embedding server:")
        print(_url_open_no_proxy(f"{emb_base}/models"))
    if rerank_base is not None:
        print("\n📋 Modelos en Reranker server:")
        print(_url_open_no_proxy(f"{rerank_base}/models"))

    if set_env:
        os.environ["VLLM_LLM_API_BASE"] = llm_base
        os.environ["VLLM_API_BASE"] = llm_base
        if emb_base is not None:
            os.environ["VLLM_EMB_API_BASE"] = emb_base
        if rerank_base is not None:
            os.environ["VLLM_RERANK_API_BASE"] = rerank_base

        # Clave dummy: vLLM ignora el valor, sólo requiere que exista.
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
        os.environ["LLM_SERVED_NAME"] = cfg.llm_served_name
        os.environ["EMB_SERVED_NAME"] = cfg.emb_served_name
        os.environ["RERANK_SERVED_NAME"] = cfg.rerank_served_name

        print("\n🌐 Bases URL registradas en ENV:")
        print("VLLM_API_BASE       =", os.environ["VLLM_API_BASE"])
        print("VLLM_LLM_API_BASE   =", os.environ["VLLM_LLM_API_BASE"])
        if emb_base is not None:
            print("VLLM_EMB_API_BASE   =", os.environ["VLLM_EMB_API_BASE"])
        if rerank_base is not None:
            print("VLLM_RERANK_API_BASE=", os.environ["VLLM_RERANK_API_BASE"])

    endpoints = VllmEndpoints(
        llm_base_url=llm_base,
        emb_base_url=emb_base,
        rerank_base_url=rerank_base,
    )
    servers = VllmServers(
        llm_proc=llm_proc,
        emb_proc=emb_proc,
        rerank_proc=rerank_proc,
        llm_log_path=llm_log,
        emb_log_path=emb_log,
        rerank_log_path=rerank_log,
    )
    return endpoints, servers


# ─────────────────────────────────────────────
# PlannerConfig + planner LLM
# ─────────────────────────────────────────────

STRICT_SYSTEM_TEXT = (
    "Eres el PLANNER de herramientas de un agente de IA.\n"
    "Tu trabajo es LEER con cuidado la petición completa del usuario, "
    "descomponerla en subtareas cuando sea necesario y planificar una o "
    "VARIAS llamadas a herramientas para resolver TODAS las partes.\n\n"
    "Dispones de un conjunto de herramientas (tools). Para cada tool "
    "conoces su nombre, su descripción y sus parámetros.\n\n"
    "INSTRUCCIONES (agnósticas de dominio):\n"
    "1) Usa SIEMPRE herramientas cuando exista alguna relevante.\n"
    "2) Devuelves únicamente tool_calls (llamadas a herramientas), "
    "   never a final answer in natural language.\n"
    "3) Si la instrucción tiene varias acciones (por ejemplo: "
    "   'primero A, luego B, al final C'), planifica todas las tools "
    "   necesarias respetando ese orden.\n"
    "4) Si la entrada menciona varios ítems (varios textos, documentos, "
    "   números, entidades, etc.), considera planear llamadas que cubran "
    "   CADA ítem relevante.\n"
    "5) Siempre que haya al menos una herramienta relevante, llama a UNA "
    "   O MÁS herramientas; no respondas sólo con razonamiento interno.\n"
    "6) Sé explícito y coherente en los argumentos de cada tool_call; "
    "   respeta nombres y tipos de parámetros.\n"
    "7) Si una acción requiere varios pasos, divide el trabajo "
    "   en varias tool_calls encadenadas.\n"
    "8) Decide qué herramientas usar únicamente por su descripción.\n"
)

FREE_SYSTEM_TEXT = (
    "Eres el asistente/PLANNER de un agente.\n"
    "Regla principal:\n"
    "- Si el usuario pide conversación, creatividad (poesía, historias), identidad del asistente, "
    "  explicación general o razonamiento que NO requiere datos externos, responde DIRECTO en lenguaje natural.\n"
    "- Usa herramientas SOLO cuando sean necesarias (DB, archivos, web, embeddings, cálculos deterministas, etc.).\n\n"
    "Cuando uses herramientas:\n"
    "- Genera tool_calls correctas (nombre + argumentos) para resolver lo pedido.\n\n"
    "Cuando NO uses herramientas:\n"
    "- NO inventes que necesitas tools.\n"
    "- Devuelve una respuesta final clara y útil.\n"
)

HYBRID_SYSTEM_TEXT = (
    "Eres un asistente IA híbrido: experto conversacional y técnico a la vez.\n\n"
    "TU FILOSOFÍA:\n"
    "1. **Proactividad con Herramientas:** Si el usuario pide algo que requiere datos o cálculo (buscar en PDF, sumar, etc.), "
    "   USA las tools de inmediato. No preguntes, actúa.\n"
    "2. **Conversación Natural:** SIEMPRE finaliza tu ejecución con una respuesta en lenguaje natural útil y amable. "
    "   No devuelvas solo el resultado crudo de la herramienta, explícalo.\n"
    "3. **Charla:** Si el usuario solo quiere charlar o ser creativo, responde con naturalidad sin forzar herramientas.\n\n"
    "INSTRUCCIONES CLAVE:\n"
    "- Devuelves `tool_calls` cuando hay trabajo técnico que hacer.\n"
    "- Cuando tengas el resultado de las tools, sintetízalo en una respuesta final clara (final_answer)."
)


@dataclass
class PlannerConfig:
    """Configuración de alto nivel del planner de herramientas."""
    model_name: str = os.getenv("LLM_SERVED_NAME", "qwen3-0.6b")
    tool_choice: str = os.getenv("PLANNER_TOOL_CHOICE", "required")
    max_retries: int = int(os.getenv("PLANNER_MAX_RETRIES", "1"))
    max_steps: int = int(os.getenv("PLANNER_MAX_STEPS", "16"))
    temperature: float = float(os.getenv("PLANNER_TEMPERATURE", "0.0"))

    # Reasoning
    enable_thinking: bool = _str_to_bool(os.getenv("PLANNER_ENABLE_THINKING", "true"))

    # Timeout para evitar httpx.ReadTimeout
    request_timeout: float = float(os.getenv("PLANNER_REQUEST_TIMEOUT", "120.0"))

    # Prompt base del planner (agnóstico de dominio)
    # Se complementa con el Unified Registry injected en logic.py
    system_text: str = (
        "Eres el PLANNER de un agente de IA con acceso a un Registro de Capacidades Unificado.\n"
        "Tu trabajo es LEER con cuidado la petición del usuario y orquestar una solución usando "
        "las Tools, Knowledge y Skills disponibles.\n\n"
        "INSTRUCCIONES GENERALES:\n"
        "1) Si una Skill aplica (ver descripción), ÚSALA prioridad máxima.\n"
        "2) Si el usuario pide datos específicos (HECHOS, PERSONAS, ENTIDADES), debés usar herramientas de BÚSQUEDA (Search / Knowledge).\n"
        "   - JAMÁS inventes que necesitas calcular matemáticas para una pregunta de historia o texto.\n"
        "   - Si no sabes qué tool usar para un dato, usa la tool de búsqueda más genérica disponible.\n"
        "3) Si necesitas transformar datos (cálculos, formato), usa Tools.\n"
        "4) Sé explícito en los argumentos de cada tool_call.\n"
        "5) No asumas valores que no tienes; busca o pregunta.\n"
    )


def build_planner_system_message(
    config: Optional[PlannerConfig] = None,
) -> SystemMessage:
    """Devuelve el SystemMessage base para el Planner (Agnóstico)."""
    cfg = config or PlannerConfig()
    return SystemMessage(content=cfg.system_text)

def build_planner_llm(config: Optional[PlannerConfig] = None) -> Any:
    """
    Construye el LLM planner apuntando al endpoint vLLM (VLLM_API_BASE).
    No asume tools; se configuran fuera con .bind_tools().
    """
    cfg = config or PlannerConfig()

    # Compatibilidad: si solo se definió VLLM_LLM_API_BASE, lo usamos como VLLM_API_BASE.
    if "VLLM_LLM_API_BASE" in os.environ and "VLLM_API_BASE" not in os.environ:
        os.environ["VLLM_API_BASE"] = os.environ["VLLM_LLM_API_BASE"]

    if HAS_LANGCHAIN_QWQ:
        return ChatGenVllm(
            model=cfg.model_name,
            temperature=cfg.temperature,
            enable_thinking=cfg.enable_thinking,
            request_timeout=cfg.request_timeout,
            openai_api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    # Fallback for environments without langchain_qwq (e.g. local/OpenAI stack).
    return ChatGenVllm(
        model=cfg.model_name,
        temperature=cfg.temperature,
        timeout=cfg.request_timeout,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("VLLM_API_BASE"),
    )
