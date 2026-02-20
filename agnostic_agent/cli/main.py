import argparse
import os

from agnostic_agent.agent import Agent
from agnostic_agent.capabilities import PlannerConfig
from agnostic_agent.config.loader import settings, load_config
from agnostic_agent.config.overrides import override_settings

def parse_args():
    parser = argparse.ArgumentParser(description="Agnostic Agent CLI")
    parser.add_argument("--prompt", "-p", type=str, help="User prompt to execute")
    parser.add_argument("--profile", type=str, default="dev", help="Configuration profile (dev/prod)")
    parser.add_argument("--config", type=str, help="Path to config directory")
    parser.add_argument("--session-id", type=str, default="cli_session", help="Session ID for memory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    # Overrides
    parser.add_argument("--provider", type=str, help="Override LLM provider")
    parser.add_argument("--model", type=str, help="Override LLM model")
    
    return parser.parse_args()

def run_once(agent: Agent, prompt: str, session_id: str):
    print(f"🤖 User: {prompt}")
    try:
        result = agent.run_turn({
            "user_prompt": prompt,
            "session_id": session_id
        })
        user_out = result.get("user_out", {}).get("final_answer", "")
        print(f"🤖 Agent: {user_out}")
        
        if settings.debug:
            print(f"\n[DEBUG] Tool Runs: {len(result.get('dev_out', {}).get('tool_runs', []))}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()

def main():
    args = parse_args()
    
    # Load Config
    if args.config:
        os.environ["AGNOSTIC_CONFIG_DIR"] = args.config
        
    if args.profile:
        os.environ["AGNOSTIC_PROFILE"] = args.profile
    
    # Apply overrides via context manager or environment
    overrides = {}
    if args.provider:
        overrides["llm"] = {"provider": args.provider}
    if args.model:
        if "llm" not in overrides: overrides["llm"] = {}
        overrides["llm"]["model"] = args.model
        
    with override_settings(overrides):
        # Initialize Agent
        try:
            runtime_settings = load_config()

            if runtime_settings.llm.base_url:
                os.environ["VLLM_API_BASE"] = runtime_settings.llm.base_url
                os.environ["VLLM_LLM_API_BASE"] = runtime_settings.llm.base_url

            planner_cfg = PlannerConfig(
                model_name=os.getenv("LLM_SERVED_NAME", runtime_settings.llm.model),
                temperature=runtime_settings.llm.temperature,
            )

            agent = Agent.init(config_or_setup=planner_cfg)
            
            if args.interactive:
                print(f"Starting interactive session (id: {args.session_id}). Type 'exit' to quit.")
                while True:
                    try:
                        user_input = input("You: ")
                        if user_input.lower() in ("exit", "quit"):
                            break
                        run_once(agent, user_input, args.session_id)
                    except KeyboardInterrupt:
                        break
            elif args.prompt:
                run_once(agent, args.prompt, args.session_id)
            else:
                print("Please provide a prompt with --prompt or use --interactive")
                
        except Exception as e:
            print(f"Failed to initialize agent: {e}")
            if settings.debug:
                raise

if __name__ == "__main__":
    main()
