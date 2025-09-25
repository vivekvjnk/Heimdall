from langchain.output_parsers import YamlOutputParser
from typing_extensions import override
import json
from .yaml_format_instructions import yaml_format_instructions


class PstYamlOutputParser(YamlOutputParser):
    @override
    def get_format_instructions(self,few_shot_examples=True) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = {k: v for k, v in self.pydantic_object.schema().items()}

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure yaml in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return yaml_format_instructions(schema=schema_str,few_shot_examples=few_shot_examples)