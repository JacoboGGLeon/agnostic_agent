
import json
import uuid
import sys
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

# Ajustar path si es necesario
sys.path.append(".")

from agnostic_agent.logic import load_logic
from agnostic_agent.schemas import AgentState

# --- Mocking ---

class MockTool:
    def __init__(self, name: str):
        self.name = name
    
    def invoke(self, args: Dict[str, Any]) -> Any:
        print(f"    [TOOL EXECUTING] {self.name} with args={args}")
        if self.name == "embed_texts":
            return [[0.01, 0.02, 0.03]] * len(args.get("texts", []))
        return "mock_success"

class MockLLM(Runnable):
    def invoke(self, input: Any, config: Any = None) -> AIMessage:
        print("\n  [LLM INVOKED]")
        
        # Debug del contenido del último mensaje
        last_msg = ""
        if isinstance(input, list):
            last_msg = str(input[-1].content)
        else:
            last_msg = str(input)
        
        # print(f"    Input prompt snippet: {last_msg[:100]}...")

        # 1. ANALYZER catch
        if "PROPOSICIONES ATÓMICAS" in str(input[0].content if isinstance(input, list) else ""):
            print("    -> Returning ANALYZER intent")
            return AIMessage(content="""
            ```json
            {
              "propositions": [
                {
                  "id": "P1",
                  "text": "guardar en embeddings el texto 'abracadabra...'",
                  "confidence": 1.0
                }
              ],
              "main_objective": "Test embedding",
              "language": "es"
            }
            ```
            """)

        # 2. PLANNER catch
        if "ÁRBOL DE TAREAS" in str(input[0].content if isinstance(input, list) else ""):
            print("    -> Returning PLANNER plan")
            return AIMessage(content="""
            ```json
            {
              "tasks": [
                {
                  "id": "T1",
                  "instruction": "Embed 'abracadabra'",
                  "dependencies": [],
                  "tool_name": "embed_texts",
                  "tool_args": {"texts": ["abracadabra_content"]}
                }
              ],
              "rationale": "Just one step"
            }
            ```
            """)

        # 3. SUMMARIZER catch
        if "Genera la respuesta final" in last_msg:
            print("    -> Returning FINAL ANSWER")
            return AIMessage(content="He ejecutado la tarea correctamente.")

        print("    -> Returning UNKNOWN Fallback")
        return AIMessage(content="I am confused.")

# --- Main ---

def main():
    print(">>> STARTING VERIFICATION SCRIPT")
    
    mock_llm = MockLLM()
    tools = [MockTool("embed_texts")]
    
    print(">>> Loading Logic Graph...")
    app = load_logic(mock_llm, tools)
    print(">>> Logic Graph Loaded.")
    
    prompt = "puedes guardar en embeddings el siguiente texto: 'abracadabra'..."
    state = {"user_prompt": prompt}
    
    print(f">>> Invoking Graph with prompt: {prompt[:30]}...")
    
    # Usamos stream para ver paso a paso si es necesario, 
    # pero invoke es más directo. Si se cuelga, veremos el último print.
    try:
        final_state = app.invoke(state, {"recursion_limit": 20})
        print(">>> Graph Invocation Finished.")
    except Exception as e:
        print(f"\n>>> EXCEPTION DURING INVOKE: {e}")
        return

    print("\n>>> INSPECTING RESULTS:")
    
    intent = final_state.get("analyzer_intent")
    print(f"  Analyzer intent present: {intent is not None}")
    if intent:
        print(f"  Propositions: {len(intent.propositions)}")

    plan = final_state.get("planner_plan")
    print(f"  Planner plan present: {plan is not None}")
    if plan:
        print(f"  Tasks: {len(plan.tasks)}")
        if plan.tasks:
            t1 = plan.tasks[0]
            print(f"  Task T1 Status: {t1.status}")
            print(f"  Task T1 Result: {t1.result}")

    print(f"  Final Output: {final_state.get('user_out', 'N/A')}")
    print(">>> VERIFICATION DONE.")

if __name__ == "__main__":
    main()
