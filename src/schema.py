from pydantic import BaseModel, Field
from typing import List, Optional

class ContactInfo(BaseModel):
    emails: List[str] = Field(default_factory=list, description="Email addresses")
    phones: List[str] = Field(default_factory=list, description="Phone numbers")

class PersonalInfo(BaseModel):
    fullName: str = Field(description="Full name of the candidate, including any titles used, such as Dr, Professor, etc.")
    personalStatement: Optional[str] = Field(None, description="Introductory statement, profile summary, or personal description")
    contact: Optional[ContactInfo] = Field(None, description="Contact info containing emails and phone numbers")

class EducationItem(BaseModel):
    institution: Optional[str] = Field(None, description="Name of the institution")
    qualificationTitle: Optional[str] = Field(None, description="Name of the academic degree or qualification earned, such as 'Mechanical Engineering, PhD', etc.")
    startDate: Optional[str] = Field(None, description="Start date or year of enrollment (as a string)")
    endDate: Optional[str] = Field(None, description="End date or year of graduation, or null if current (as a string)")
    description: Optional[str] = Field(None, description="Any additional information about the education, such as GPA, honors, mentors, coursework, etc.")

class WorkExperienceItem(BaseModel):
    employer: Optional[str] = Field(None, description="Employer name")
    jobTitle: Optional[str] = Field(None, description="Job title/position")
    startDate: Optional[str] = Field(None, description="Start date/year (as a string)")
    endDate: Optional[str] = Field(None, description="End date/year, or null if current (as a string)")
    isCurrent: Optional[bool] = Field(None, description="Whether this is current position")
    category: Optional[str] = Field(None, description="Type of experience (e.g., 'Teaching Experience', 'Research Experience', 'Industry Experience', 'Administrative Experience').")
    description: Optional[str] = Field(None, description="Description or responsibilities for the role")

class PublicationItem(BaseModel):
    link: Optional[str] = Field(None, description="URL to the publication")
    year: Optional[str] = Field(None, description="Publication year")
    title: Optional[str] = Field(None, description="Title of the publication")
    authors: Optional[str] = Field(None, description="Author name(s) as they appear in the publication")
    publisher: Optional[str] = Field(None, description="Journal or publisher of the publication")

class CertificationAwardItem(BaseModel):
    date: Optional[str] = Field(None, description="Date when credential was awarded or obtained")
    category: Optional[str] = Field(None, description="Type of credential (e.g., 'Certification', 'Award', 'Membership', 'License', 'Honor', 'Affiliation').")
    description: Optional[str] = Field(None, description="Description of the certification, award, membership, etc.")
    organization: Optional[str] = Field(None, description="Organization associated with the credential")

class OtherItem(BaseModel):
    content: Optional[str] = Field(None, description="The content or description for this miscellaneous section")
    sectionTitle: Optional[str] = Field(None, description="Miscellaneous sections")

class SkillGroup(BaseModel):
    category: str = Field(description="Skill category name (e.g., 'Technical Skills', 'Soft Skills', or 'General')")
    items: List[str] = Field(description="List of skills or technical competencies in this category")

class ResumeSchema(BaseModel):
    personalInfo: PersonalInfo = Field(description="Personal information of the candidate")
    workExperience: List[WorkExperienceItem] = Field(default_factory=list, description="Work experience history. For academic resumes, this should include teaching experience, research experience, academic appointments and other relevant experience.")
    education: List[EducationItem] = Field(default_factory=list, description="Education history")
    skills: List[SkillGroup] = Field(default_factory=list, description="List of skills or competencies grouped by category")
    languages: List[str] = Field(default_factory=list, description="Languages spoken and proficiency levels")
    socialLinks: List[str] = Field(default_factory=list, description="Links to professional profiles such as LinkedIn, GitHub, personal websites")
    publications: List[PublicationItem] = Field(default_factory=list, description="Academic publications, research papers, articles, or similar")
    certificationsAndAwards: List[CertificationAwardItem] = Field(default_factory=list, description="Professional certifications, licenses, awards, honors, memberships, and affiliations")
    media: List[str] = Field(default_factory=list, description="Media mentions, appearances, or similar")
    other: List[OtherItem] = Field(default_factory=list, description="Catch-all for miscellaneous sections not captured elsewhere.")


import json
import os
from pydantic import create_model
from typing import Dict, Any, Tuple, Type

def resolve_schema_node(node: Any, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves $ref and any nesting reference paths in JSON schema."""
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        ref_path = node["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path.split("/")[1:]
            current = root_schema
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    raise ValueError(f"Could not resolve reference {ref_path} in schema")
            return resolve_schema_node(current, root_schema)
    return node

def json_node_to_python_type(node: Dict[str, Any], root_schema: Dict[str, Any], prop_name: str) -> Tuple[Type, Optional[str]]:
    """Converts a JSON Schema node to a Pydantic-compatible Python type, resolving unions."""
    node = resolve_schema_node(node, root_schema)
    description = node.get("description", None)
    
    if "anyOf" in node or "oneOf" in node:
        options = node.get("anyOf") or node.get("oneOf")
        non_null_options = []
        has_null = False
        for opt in options:
            opt_resolved = resolve_schema_node(opt, root_schema)
            if opt_resolved.get("type") == "null":
                has_null = True
            else:
                non_null_options.append(opt_resolved)
                
        if not non_null_options:
            return Optional[Any], description

        # If the union mixes number/integer with string (e.g. value field), collapse to str
        # to avoid Gemini schema state explosion from numeric constraints
        types_in_union = {o.get("type") for o in non_null_options}
        if types_in_union & {"number", "integer", "string"} and len(types_in_union) > 1:
            return (Optional[str] if has_null else str), description

        # Select first non-null type to satisfy Gemini API constraints (no Union/anyOf)
        primary_opt = non_null_options[0]
        primary_type, primary_desc = json_node_to_python_type(primary_opt, root_schema, prop_name)
        if description is None:
            description = primary_desc
            
        if has_null:
            return Optional[primary_type], description
        return primary_type, description
        
    json_type = node.get("type", "string")
    
    if json_type == "string":
        return str, description
    elif json_type == "integer":
        # Use str instead of int to avoid Gemini numeric constraint state explosion
        return str, description
    elif json_type == "number":
        # Use str instead of float to avoid Gemini numeric constraint state explosion
        return str, description
    elif json_type == "boolean":
        return bool, description
    elif json_type == "array":
        items_node = node.get("items")
        if items_node:
            item_type, item_desc = json_node_to_python_type(items_node, root_schema, f"{prop_name}_item")
            return List[item_type], description
        else:
            return List[Any], description
    elif json_type == "object":
        nested_model = build_pydantic_model_from_json_schema(node, root_schema, name=f"{prop_name.capitalize()}SubModel")
        return nested_model, description
    else:
        return Any, description

def build_pydantic_model_from_json_schema(schema_dict: Dict[str, Any], root_schema: Optional[Dict[str, Any]] = None, name: str = "DynamicModel") -> Type[BaseModel]:
    """Recursively creates a Pydantic BaseModel from JSON Schema."""
    if root_schema is None:
        root_schema = schema_dict
        
    schema_dict = resolve_schema_node(schema_dict, root_schema)
    
    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])
    
    fields = {}
    for prop_name, prop_val in properties.items():
        prop_val = resolve_schema_node(prop_val, root_schema)
        # Strip regex pattern and string format constraints — they cause Gemini state explosion
        prop_val = {k: v for k, v in prop_val.items() if k not in ("pattern", "format")}
        py_type, description = json_node_to_python_type(prop_val, root_schema, prop_name)
        
        is_required = prop_name in required
        if is_required:
            field_def = Field(..., description=description)
        else:
            default_val = prop_val.get("default", None)
            field_def = Field(default=default_val, description=description)
            
        fields[prop_name] = (py_type, field_def)
        
    return create_model(name, **fields)

def locate_schema_file(schema_name: str) -> str:
    """Searches dataset/ directory for the JSON schema file."""
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file.lower() == f"{schema_name}-schema.json" or file.lower() == f"{schema_name}_schema.json":
                return os.path.join(root, file)
    raise FileNotFoundError(f"Could not find schema file matching {schema_name} inside dataset/")

def load_schema_dict(schema_name: str) -> Dict[str, Any]:
    """Loads the raw schema dictionary from disk."""
    schema_file = locate_schema_file(schema_name)
    with open(schema_file, "r") as f:
        schema_dict = json.load(f)
    if "schema_definition" in schema_dict:
        schema_dict = schema_dict["schema_definition"]
    return schema_dict

def get_schema_class(schema_name: str) -> Type[BaseModel]:
    """Returns custom ResumeSchema for compatibility or compiles dynamic Pydantic model."""
    if schema_name.lower() == "resume":
        return ResumeSchema
    schema_dict = load_schema_dict(schema_name)
    model_name = schema_name.replace("_", " ").replace("-", " ").title().replace(" ", "") + "Schema"
    return build_pydantic_model_from_json_schema(schema_dict, name=model_name)

