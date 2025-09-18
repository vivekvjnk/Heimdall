CORRECTION_PROMPT = """
The previous output had a formatting issue. Please correct it and return a valid YAML output.

===Original Output Begin===
{original_output}
===Original Output End===

===Error Description Begin===
{error_description}
===Error Description End===

Your response must be a single, valid YAML block that conforms to the format instructions.
===FORMAT INSTRUCTIONS BEGIN===
{format_instructions}
===FORMAT INSTRUCTIONS END===

Corrected YAML Output:
"""

ERROR_REFLECTION_PROMPT= """
Following is the pydantic basemodel used for parsing llm output:
===Basemodel Begin===
{base_model}
===Basemodel End===

The LLM output(in yml format) had the following structuring issue:

===Structuring Issue Begin===
{yaml_format_issues}
===Structuring Issue End===

Please analyze the issue and provide a brief passage describing:

1. The root cause of the issue. Specifically analyze if the issue can be solved by quoting the values.
2. A suggested solution in clear terms.
3. The exact location of the problem.
Additionally, suggest to enclose every string in quotes. Respond within 100 words.

Error reflections:
"""
