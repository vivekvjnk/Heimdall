few_shot_examples = """# Examples
## Schema
```
{{"title": "Players", "description": "A list of players", "type": "array", "items": {{"$ref": "#/definitions/Player"}}, "definitions": {{"Player": {{"title": "Player", "type": "object", "properties": {{"name": {{"title": "Name", "description": "Player name", "type": "string"}}, "avg": {{"title": "Avg", "description": "Batting average", "type": "number"}}}}, "required": ["name", "avg"]}}}}}}
```
## Well formatted instance
```
- name: John Doe
  avg: 0.3
- name: Jane Maxfield
  avg: 1.4
```

## Schema
```
{{"properties": {{"habit": {{ "description": "A common daily habit", "type": "string" }}, "sustainable_alternative": {{ "description": "An environmentally friendly alternative to the habit", "type": "string"}}}}, "required": ["habit", "sustainable_alternative"]}}
```
## Well formatted instance
```
habit: Using disposable water bottles for daily hydration.
sustainable_alternative: Switch to a reusable water bottle to reduce plastic waste and decrease your environmental footprint.
```
"""

general_instructions = """The output should be formatted as a YAML instance that conforms to the given JSON schema below."""

user_schema_instructions = """Please follow the standard YAML formatting conventions with an indent of 2 spaces and make sure that the data types adhere strictly to the following JSON schema:
```
{schema}
```
"""

additional_instructions = """Make sure to always enclose the YAML output in triple backticks (```). Please do not add anything other than valid YAML output! Avoid using YAML formatting characters such as the mapping character (:) within string values. Enclose all strings and sentences in double quotes ("")."""

def yaml_format_instructions(schema,few_shot_examples=True):
    
    user_schema = user_schema_instructions.format(schema=schema)
    if few_shot_examples:
        YAML_FORMAT_INSTRUCTIONS = f"{general_instructions}\n{few_shot_examples}\n{user_schema}\n{additional_instructions}"
    else:
        YAML_FORMAT_INSTRUCTIONS = f"{general_instructions}\n{user_schema}\n{additional_instructions}"
        
    return YAML_FORMAT_INSTRUCTIONS
