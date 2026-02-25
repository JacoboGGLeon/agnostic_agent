import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from agnostic_agent.agent import Agent
from agnostic_agent.capabilities import PlannerConfig
from agnostic_agent.api.models import ChatRequest, ChatResponse, SkillInfo, ToolInfo, SettingsResponse
from agnostic_agent.config.loader import load_config
from agnostic_agent.plugins.manager import PluginManager

app = FastAPI(title="Agnostic Agent API", version="1.0.0")

# Enable CORS for local Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent Singleton
_agent_instance: Agent | None = None
_plugin_manager: PluginManager | None = None

def get_agent(cfg: PlannerConfig = None) -> Agent:
    global _agent_instance, _plugin_manager
    
    if _plugin_manager is None:
        try:
            config = load_config()
            _plugin_manager = PluginManager(config.plugins.model_dump())
            _plugin_manager.load_plugins()
        except Exception as e:
            print(f"Warning: Failed to load plugins: {e}")

    if _agent_instance is None:
        if cfg is None:
            # Fallback default cfg
            cfg = PlannerConfig(
                model_name=os.getenv("LLM_SERVED_NAME", "deepseek-coder"),
                temperature=0.0,
                max_steps=15
            )
        _agent_instance = Agent.init(config_or_setup=cfg)
    return _agent_instance


@app.get("/api/v1/capabilities/skills", response_model=List[SkillInfo])
async def get_skills():
    """Returns the list of available skills from the registry"""
    agent = get_agent()
    skills_list = []
    
    if agent.skill_registry:
        for s_name, skill in agent.skill_registry.skills.items():
            skills_list.append(SkillInfo(
                name=s_name,
                description=skill.description,
                enabled=agent.skill_registry.is_enabled(s_name),
                tools=skill.tools or []
            ))
            
    return skills_list

@app.get("/api/v1/settings", response_model=SettingsResponse)
async def get_settings():
    """Returns the default Server LLM and Embedding Models configured by the Environment Variables"""
    # Logic matching v94 sidebar.py resolution
    llm_name = os.getenv("LLM_SERVED_NAME") or os.getenv("OPENAI_MODEL") or os.getenv("AGNOSTIC_LLM_MODEL") or "custom-llm-model"
    emb_name = os.getenv("EMB_SERVED_NAME") or os.getenv("OPENAI_EMBED_MODEL") or os.getenv("AGNOSTIC_EMB_MODEL") or "custom-embed-model"
    
    return SettingsResponse(
        llm_served_name=llm_name,
        emb_served_name=emb_name
    )



@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main endpoint to invoke the Agentic LangGraph pipeline"""
    
    # 1. Update Agent Config
    planner_cfg = None
    if request.llm_config:
        planner_cfg = PlannerConfig(
            model_name=request.llm_config.model_name or os.getenv("LLM_SERVED_NAME"),
            temperature=request.llm_config.temperature,
            max_steps=request.llm_config.max_steps
        )
    
    agent = get_agent(planner_cfg)

    # 2. Sync Skills
    if agent.skill_registry and request.skills_config:
        for sname, enabled in request.skills_config.items():
            agent.skill_registry.set_enabled(sname, enabled)

    # 3. Process invoke
    try:
        # We pass the prompt. In the future we can pass request.messages_history into thread history
        result = agent.invoke(
            prompt=request.message,
            thread_id=request.session_id
        )
        
        # 4. Extract outputs
        final_answer = result.get("final_answer", "No answer generated.")
        raw_state = result.get("raw_state", {})
        
        # Build JSON metadata for Deep view
        deep_json = {
            "analyzer": {
                "subqueries": raw_state.get("analyzer", {}).get("subqueries", []),
                "active_skills": list(raw_state.get("_planner_scope_internal", {}).get("active_skills", [])),
                "allowed_tools": raw_state.get("_planner_scope_internal", {}).get("allowed_tools", []),
            },
            "planner": [
                {
                    "subquery": t.subquery if hasattr(t, 'subquery') else t.get('subquery', ''),
                    "description": t.description if hasattr(t, 'description') else t.get('description', '')
                } for t in (raw_state.get("planner_trajs") or [])
            ],
            "executor": {
                "tool_runs": [
                    {
                        "tool_name": run.tool_name if hasattr(run, 'tool_name') else run.get('tool_name', ''),
                        "args": run.args if hasattr(run, 'args') else run.get('args', {}),
                        "result": run.result if hasattr(run, 'result') else run.get('result', ''),
                        "error": run.error if hasattr(run, 'error') else run.get('error', None)
                    } for run in (raw_state.get("tool_runs") or [])
                ]
            },
            "summarizer": raw_state.get("summarizer_out", "")
        }

        return ChatResponse(
            session_id=request.session_id,
            answer=final_answer,
            deep_json=deep_json
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# Static Frontend Serving (React/Vite SPA)
# ---------------------------------------------------------
webapp_dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "webapp", "dist"))

if os.path.exists(webapp_dist_path):
    print(f"📦 Serving React Frontend natively from: {webapp_dist_path}")
    
    # Mount assets folder explicitly so static CSS/JS load correctly
    assets_path = os.path.join(webapp_dist_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        
    # Vite also sometimes creates a public folder or copies icons at root. Fallback catcher:
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Ignore calls directly to the api that were not caught
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API Route Not Found")
            
        file_path = os.path.join(webapp_dist_path, full_path)
        if full_path != "" and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # Fallback to index.html for SPA typical routing
        return FileResponse(os.path.join(webapp_dist_path, "index.html"))
else:
    print(f"⚠️ Warning: Frontend build not found at {webapp_dist_path}. Run 'npm run build' inside 'webapp/' first.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agnostic_agent.api.server:app", host="127.0.0.1", port=8000, reload=True)
