from setuptools import setup, find_packages

setup(
    name="agnostic_agent",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "typing_extensions",
        "pydantic>=2.7.0",
        "pyyaml>=6.0",
        "streamlit",
        "sqlite-vec",
        "pandas",
        "numpy",
        "pymupdf",          # for PDF parsing (fitz)
        "transformers",     # for local embeddings
        "torch",            # for local embeddings
        "openai",           # for potential vLLM client
    ],
)
