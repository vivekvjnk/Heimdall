[//]: # (This template is designed to capture invariants, variables, and ergonomics of repeated tasks.)
[//]: # (It helps us distill what always stays the same, what changes, and how to make tasks smoother.)

# Task: Convert validationwrapper into publishable langgraph graph
_Date: 2025-09-18_

---

## 1. 🦴 Skeleton (Always present backbone)
[//]: # (List the elements that never change. The backbone of the task.)
- **Inputs**  
  - List the key inputs: configs, datasets, external files
- **Process**  
  1. Identify all dependencies of the module. Extend of coupling of the module with the current code base.
  2. Conversion of the module into langgraph runnable graph. 
    - ValidationWrapper and many other modules in the system are designed as dedicated classes with nodes as member functions. Langgrpah expect simple functions without any class wraps. Hence we need to convert class implementations into simple functions with a graph construction function.
  3. Convert the module to a pip package, aligning with langgraph subgraph requirements
- **Outputs**  
  - *List the artifacts or results always produced*
  - Runnable langgraph graph with clear inputs and outputs 
---

## 2. 🎛️ Variables (What changes between runs?)
[//]: # (Capture the parameters or environment-dependent elements.)
- **Parameters**  
  - Config values, thresholds, tunable knobs
- **Artifacts**  
  - Different entities, files, graphs produced
- **Environments**  
  - LLM version, dataset split, GPU seed, etc.

---

## 3. 🛠️ Ergonomics (How to make it smoother next time?)
[//]: # (Tricks, shortcuts, or reusable resources to reduce friction.)
- **Shortcuts**  
  - Time-savers, scripts, aliases
- **Templates**  
  - Reusable boilerplate: configs, prompt skeletons
- **Checks**  
  - Sanity checks, asserts, validation rules

---

## 🔍 Observations
[//]: # (Optional free space to write insights, patterns, or invariants you discovered.)
- Optional: Notes on patterns or invariants discovered here