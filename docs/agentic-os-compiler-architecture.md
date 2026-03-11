# Agentic OS as Compiler Architecture

## Purpose

This note captures a cleaner architectural framing for a future iteration of `agnostic_agent`.

Instead of describing the system as "a chatbot with tools", this framing treats it as a
compiler-and-runtime pipeline for skill-world execution:

`natural language -> semantic frontend -> propositional IR -> planner/compiler -> runtime -> renderer`

The goal is to make the system easier to reason about, extend, audit, and evolve toward an
Agentic OS runtime.

## Core Idea

The system should not be modeled as a monolithic agent improvising over tool calls.

It should be modeled as a pipeline with clearly separated responsibilities:

1. A semantic frontend interprets the user request.
2. An intermediate representation captures propositions, entities, coverage, and dependencies.
3. A planner/compiler turns each proposition into an execution DAG.
4. A runtime executes DAG nodes and collects artifacts.
5. A renderer produces user/dev/deep responses from verified evidence.

This is closer to a compiler architecture than to a generic assistant loop.

## Architectural Layers

### 1. Semantic Frontend

The semantic frontend receives raw natural language and turns it into structured meaning.

Responsibilities:

- detect the active skill-world
- identify entities
- infer constraints
- infer coverage expectations
- split the request into propositions or subqueries
- identify logical relations between propositions

In the current system, this role is primarily performed by the analyzer.

Examples:

- `dame informacion sobre el credito LOC-0004`
- `estos creditos`
- `concilialos`
- `de tus bases de datos`

The frontend should normalize these into explicit semantic units rather than passing the raw
request downstream unchanged.

### 2. Propositional IR

The Propositional IR is the system's intermediate representation between language and execution.

It should contain information such as:

- propositions: `q1`, `q2`, `q3`, ...
- propositional logic: `q1 AND q2`
- dependencies: `q3 depends_on q1, q2`
- selected skill-world
- entities by proposition
- required sources by proposition
- coverage expectation
- composition mode

Example:

- `q1 = snapshot contable de LOC-0004`
- `q2 = movimientos de LOC-0004`
- `logic = q1 AND q2`
- `composition_mode = merge`

This IR should be more precise than the original prompt and more abstract than tool calls.

### 3. Planner / Compiler

The planner/compiler receives the Propositional IR and turns each proposition into an executable
plan.

Responsibilities:

- compile each proposition into a DAG
- select tools or workflows
- define node dependencies
- declare expected artifacts
- restrict execution to the active skill-world

Examples:

- `q1 -> nl2sql(contabilidad.db, LOC-0004)`
- `q2 -> nl2sql(transacciones.db, LOC-0004)`
- `q1 -> reconcile_credit_accounting(LOC-0004)`

This layer should not interpret the user request again. It should compile the IR into execution.

### 4. Runtime

The runtime executes the compiled plan and materializes evidence.

Responsibilities:

- execute tools and workflows
- manage state and session memory
- capture tool outputs
- normalize artifacts
- preserve provenance
- handle execution failures

In the current system, this role is distributed across executor, catcher, session memory, and
runtime/reporting components.

### 5. Renderer

The renderer produces the final outputs from verified evidence.

Responsibilities:

- generate `user_out`
- render `dev_out`
- render `deep_out`
- keep the three views aligned to the same factual bundle
- adapt aggregation level per audience

The renderer should not invent evidence or repair missing semantics upstream.

It should render what the runtime actually established.

## Mapping to Current Components

This is the current rough mapping in the repository:

- semantic frontend:
  - `agnostic_agent/tools/pipeline/analyzer_tool.py`
- propositional IR:
  - analyzer state fields such as:
    - `subqueries`
    - `propositional_logic`
    - `subquery_intents`
    - `entities_by_subquery`
    - `required_sources_by_subquery`
    - `composition_mode`
- planner/compiler:
  - `agnostic_agent/tools/pipeline/planner_tool.py`
- runtime:
  - `agnostic_agent/tools/pipeline/executor_tool.py`
  - `agnostic_agent/tools/pipeline/catcher_tool.py`
  - `agnostic_agent/app/turn_service.py`
  - `agnostic_agent/memory.py`
- renderer:
  - `agnostic_agent/tools/pipeline/summarizer_tool.py`
  - `agnostic_agent/graph/summarization.py`

## Why This Framing Matters

This framing improves the system along four dimensions:

### 1. Auditability

You can inspect:

- what the user asked
- how it was decomposed
- what propositions were compiled
- what DAGs were executed
- what evidence was collected
- how the final answer was rendered

### 2. Extensibility

New skill-worlds can reuse the same architecture:

- new semantic policies
- new IR enrichments
- new planner strategies
- new runtime capabilities

### 3. Reliability

If the system fails, it becomes possible to say whether the failure occurred in:

- semantic parsing
- proposition decomposition
- compilation
- execution
- rendering

### 4. Product Clarity

This positions the project as more than an agent shell.

It becomes plausible to describe the system as:

- an agent runtime
- a semantic execution kernel
- an Agentic OS runtime

## Future Direction

The next strong iteration should deepen the compiler model in three places.

### A. Stronger Propositional IR

Add richer structures for:

- proposition dependencies
- source scope
- batch decomposition
- coverage contracts
- coreference resolution outputs

### B. Structured Working Memory

The semantic frontend should not rely on transcript replay alone.

It should consume structured working memory such as:

- active entities
- last listed credit ids
- active sources
- last operation
- recent turn snapshots

This is especially important for:

- `estos creditos`
- `los listados`
- `concilialos`
- `dame el detalle`

### C. Batch Compilation

When a request refers to a list of active entities, the semantic frontend should emit one
proposition per entity when appropriate, and the planner/compiler should build one DAG per
proposition.

This enables proper divide-and-conquer execution for finance and other skill-worlds.

## Recommended Terminology

To keep documentation and design discussions precise:

- "semantic frontend" is better than "the analyzer understands the user"
- "propositional IR" is better than "subqueries only"
- "planner/compiler" is better than "the planner chooses tools"
- "runtime" is better than "tool execution"
- "renderer" is better than "response generator"

## One-Line Summary

The target architecture is:

`natural language -> semantic frontend -> propositional IR -> planner/compiler -> runtime -> renderer`

This framing makes `agnostic_agent` easier to evolve into an Agentic OS runtime rather than a
tool-calling chatbot.
