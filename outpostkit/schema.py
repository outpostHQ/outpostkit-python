from typing import Optional

from packaging import version


def version_has_no_array_type(inference_version: str) -> Optional[bool]:
    """Iterators have x-inference-array-type=iterator in the schema from 0.3.9 onward"""
    try:
        return version.parse(inference_version) < version.parse("0.3.9")
    except version.InvalidVersion:
        return None


def make_schema_backwards_compatible(
    schema: dict,
) -> dict:
    """A place to add backwards compatibility logic for our openapi schema"""

    # If the top-level output is an array, assume it is an iterator in old versions which didn't have an array type
    if version_has_no_array_type(inference_version):
        output = schema["components"]["schemas"]["Output"]
        if output.get("type") == "array":
            output["x-inference-array-type"] = "iterator"
    return schema
