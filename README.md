# Heimdall

**Heimdall** is a LangGraph-compatible utility for validating and correcting structured outputs from Large Language Models (LLMs) against Pydantic models. It automates error handling, correction, and validation for YAML/JSON outputs, making it easy to enforce schema compliance in LLM workflows.

## Features

- **Automated Validation:** Checks LLM outputs against Pydantic models.
- **Error Correction Loop:** Automatically reflects on and corrects formatting/validation errors.
- **Modular Design:** Easily integrates as a LangGraph subgraph or standalone validator.
- **Flexible LLM Support:** Works with any LangChain-compatible LLM (e.g., OpenAI, VertexAI, Ollama).
- **Customizable Prompts:** Uses configurable correction and reflection prompts.

## Installation

```sh
pip install heimdall
```
## Usage
```python
from heimdall import structured_output_validator, HeimdallState
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your schema
class MyData(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

# Initialize LLMs
main_llm = ChatOpenAI(model="gpt-3.5-turbo")
correction_llm = ChatOpenAI(model="gpt-4")

# Create the validator graph
validator = structured_output_validator(
    pydantic_model=MyData,
    llm=main_llm,
    thinking_model=correction_llm
)

# Prepare initial state
state = HeimdallState(
    messages=[("human", "My name is Bob and I am 30 years old. Format this.")],
    llm_output="",
    error_status=False,
    error_description="",
    iterations=0
)

# Run validation
result = validator.invoke(state)
print(result['llm_output'])  # {'name': 'Bob', 'age': 30}
```

## API
`structured_output_validator`
Creates a LangGraph graph object for structured output validation and correction.

### Arguments:

- pydantic_model: Pydantic model for output validation.
- llm: Main LLM (LangChain Runnable).
- thinking_model: (Optional) Stronger LLM for corrections.
- callbacks, trace_id, parser: (Optional) Advanced configuration.
### Returns:
A compiled LangGraph graph (Runnable) for validation.

`heimdall_graph`
Convenience wrapper for single-call validation.

## Contributing
Contributions are welcome! Please open issues or pull requests on GitHub.

## License
MIT License. See LICENSE for details.
