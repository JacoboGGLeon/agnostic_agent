import subprocess
import sys
from pathlib import Path

def test_cli_help():
    """Test that CLI help command runs without error."""
    cmd = [sys.executable, "-m", "agnostic_agent.cli.main", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Agnostic Agent CLI" in result.stdout

def test_cli_simple_prompt():
    """Test a simple prompt execution via CLI."""
    # This might fail if agent dependencies (vllm etc) are not running/mocked.
    # We should Mock the agent if possible or Expect failure but check for specific error.
    # For smoke test, we check if it starts up and tries to run.
    
    cmd = [
        sys.executable, 
        "-m", "agnostic_agent.cli.main", 
        "--prompt", "Hello", 
        "--profile", "dev"
    ]
    # We expect it might fail due to missing vllm, but return code might be non-zero
    # checking stderr for "Failed to initialize" or "Error" 
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # If it fails, it should be a graceful failure
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    
    # Smoke expectations:
    # - command line parses and executes code path (not help page)
    # - process exits gracefully (even when provider backend is unavailable)
    assert result.returncode == 0
    assert "Agnostic Agent CLI" not in result.stdout  # Should not show help
    assert (
        "Failed to initialize agent:" in result.stdout
        or "Agent:" in result.stdout
        or "Error:" in result.stdout
    )
