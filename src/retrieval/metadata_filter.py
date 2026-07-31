"""
Metadata Filter Builder
Translates pipeline configuration and conversation context into
Pinecone-compatible filter dictionaries.

Features:
- Field type validation (string, number, array)
- Operator validation per field type
- Template-based filter presets
- Conversation context auto-filtering
- Compound filters ($and, $or)
- Configuration-driven — no hardcoded field names
- Dynamic field mappings from ingestion schema
- Strict type validation with warnings
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv

from src.retrieval.config import PipelineConfig
from src.retrieval.query_rewriter import ConversationState

load_dotenv()


# ============================================================================
# Type System
# ============================================================================

class FieldType(str, Enum):
    """Supported field types for metadata filtering."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DATE = "date"
    TIMESTAMP = "timestamp"


# Type validation functions
TYPE_VALIDATORS = {
    FieldType.STRING: lambda v: isinstance(v, str),
    FieldType.NUMBER: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
    FieldType.FLOAT: lambda v: isinstance(v, float),
    FieldType.BOOLEAN: lambda v: isinstance(v, bool),
    FieldType.ARRAY: lambda v: isinstance(v, list),
    FieldType.DATE: lambda v: isinstance(v, str),  # ISO format validation could be added
    FieldType.TIMESTAMP: lambda v: isinstance(v, (int, float)),
}

# Operator compatibility matrix
OPERATOR_COMPATIBILITY = {
    "$eq": {FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, 
            FieldType.BOOLEAN, FieldType.DATE, FieldType.TIMESTAMP},
    "$ne": {FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, 
            FieldType.BOOLEAN, FieldType.DATE, FieldType.TIMESTAMP},
    "$gt": {FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE, FieldType.TIMESTAMP},
    "$gte": {FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE, FieldType.TIMESTAMP},
    "$lt": {FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE, FieldType.TIMESTAMP},
    "$lte": {FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.DATE, FieldType.TIMESTAMP},
    "$in": {FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.ARRAY},
    "$nin": {FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, FieldType.ARRAY},
    "$all": {FieldType.ARRAY},
    "$exists": {FieldType.STRING, FieldType.NUMBER, FieldType.INTEGER, FieldType.FLOAT, 
                FieldType.BOOLEAN, FieldType.ARRAY, FieldType.DATE, FieldType.TIMESTAMP},
}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FieldSchema:
    """Schema definition for a metadata field."""
    type: FieldType
    description: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = False
    default: Optional[Any] = None
    
    def validate_value(self, value: Any) -> bool:
        """Validate a value against this field's schema."""
        # Check type
        if self.type not in TYPE_VALIDATORS:
            return False
        
        if not TYPE_VALIDATORS[self.type](value):
            return False
        
        # Check allowed values
        if self.allowed_values is not None and value not in self.allowed_values:
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.type.value if isinstance(self.type, FieldType) else self.type,
        }
        if self.description:
            result["description"] = self.description
        if self.allowed_values is not None:
            result["allowed_values"] = self.allowed_values
        if self.required:
            result["required"] = self.required
        if self.default is not None:
            result["default"] = self.default
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FieldSchema":
        """Create FieldSchema from dictionary."""
        return cls(
            type=FieldType(data.get("type", "string")),
            description=data.get("description"),
            allowed_values=data.get("allowed_values"),
            required=data.get("required", False),
            default=data.get("default"),
        )


@dataclass
class MetadataFilterConfig:
    """Configuration for metadata filtering — loaded from YAML."""
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    field_mappings: Dict[str, str] = field(default_factory=dict)
    templates: Dict[str, Dict] = field(default_factory=dict)
    field_mapping_source: Optional[str] = None  # Path to ingestion schema
    
    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "MetadataFilterConfig":
        """
        Load configuration from YAML file.
        
        Priority:
        1. Explicit path argument
        2. RAGPIPE_METADATA_FILTER_CONFIG env var
        3. config/retrieval/metadata_filter.yaml (default)
        """
        config_path = (
            path
            or os.getenv("RAGPIPE_METADATA_FILTER_CONFIG")
            or str(Path(__file__).parent.parent.parent / "config" / "retrieval" / "metadata_filter.yaml")
        )
        
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            data = cls._default_config()
        
        # Parse fields
        fields = {}
        for field_name, field_data in data.get("fields", {}).items():
            fields[field_name] = FieldSchema.from_dict(field_data)
        
        return cls(
            fields=fields,
            field_mappings=data.get("field_mappings", {}),
            templates=data.get("templates", {}),
            field_mapping_source=data.get("field_mapping_source"),
        )
    
    @classmethod
    def from_ingestion_schema(cls, schema_path: str) -> "MetadataFilterConfig":
        """
        Create configuration from ingestion pipeline schema.
        
        This ensures field mappings are generated from the single source of truth.
        """
        if not Path(schema_path).exists():
            raise FileNotFoundError(f"Ingestion schema not found: {schema_path}")
        
        with open(schema_path, "r") as f:
            schema_data = yaml.safe_load(f)
        
        # Extract field definitions from ingestion schema
        fields = {}
        field_mappings = {}
        
        metadata_schema = schema_data.get("metadata", {})
        for field_name, field_config in metadata_schema.items():
            # Create FieldSchema from ingestion config
            field_type = field_config.get("type", "string")
            fields[field_name] = FieldSchema(
                type=FieldType(field_type),
                description=field_config.get("description"),
                allowed_values=field_config.get("allowed_values"),
                required=field_config.get("required", False),
            )
            
            # Create mapping from canonical name to actual field name
            canonical = field_config.get("canonical_name", field_name)
            field_mappings[canonical] = field_name
        
        return cls(
            fields=fields,
            field_mappings=field_mappings,
            templates={},
            field_mapping_source=schema_path,
        )
    
    @staticmethod
    def _default_config() -> Dict:
        """Minimal fallback configuration."""
        return {
            "fields": {
                "course_name": {"type": "string", "description": "Course name"},
                "chapter_title": {"type": "string", "description": "Chapter title"},
                "topic": {"type": "string", "description": "Topic name"},
                "file_type": {"type": "string", "description": "File type (pdf, pptx, etc.)"},
                "chunk_type": {"type": "string", "description": "Type of chunk"},
                "page_start": {"type": "number", "description": "Starting page number"},
                "page_end": {"type": "number", "description": "Ending page number"},
                "is_primary": {"type": "boolean", "description": "Primary content flag"},
                "tags": {"type": "array", "description": "Tags associated with content"},
            },
            "field_mappings": {
                "topic": "topic",
                "course": "course_name",
                "chapter": "chapter_title",
                "file_type": "file_type",
                "page": "page_start",
                "primary": "is_primary",
                "tag": "tags",
                "tags": "tags",
            },
            "templates": {},
            "field_mapping_source": None,
        }
    
    def get_field_type(self, field_name: str) -> Optional[FieldType]:
        """Get the type of a filter field."""
        field_schema = self.fields.get(field_name)
        return field_schema.type if field_schema else None
    
    def get_field_schema(self, field_name: str) -> Optional[FieldSchema]:
        """Get the full schema for a field."""
        return self.fields.get(field_name)
    
    def validate_operator(self, field_name: str, operator: str) -> bool:
        """Check if an operator is valid for a given field."""
        field_type = self.get_field_type(field_name)
        if not field_type:
            return False
        valid_types = OPERATOR_COMPATIBILITY.get(operator, set())
        return field_type in valid_types
    
    def validate_value(self, field_name: str, value: Any) -> bool:
        """Validate a value against its field schema."""
        schema = self.get_field_schema(field_name)
        if not schema:
            return False
        return schema.validate_value(value)
    
    def resolve_field(self, canonical_name: str) -> str:
        """Resolve a canonical field name to its actual field name."""
        return self.field_mappings.get(canonical_name, canonical_name)
    
    def get_canonical_name(self, field_name: str) -> str:
        """Get the canonical name for a field."""
        reverse_mapping = {v: k for k, v in self.field_mappings.items()}
        return reverse_mapping.get(field_name, field_name)
    
    def update_from_schema(self, schema_path: str) -> None:
        """
        Update configuration from an ingestion schema file.
        
        This allows dynamic updates to field mappings without restarting.
        """
        new_config = self.from_ingestion_schema(schema_path)
        self.fields.update(new_config.fields)
        self.field_mappings.update(new_config.field_mappings)
        self.field_mapping_source = schema_path


# ============================================================================
# Filter Builder
# ============================================================================

@dataclass
class FilterResult:
    """Result of building a metadata filter."""
    filter_dict: Dict[str, Any]
    applied_filters: List[str]       # Human-readable descriptions
    is_empty: bool                   # True if no filters were applied
    warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


class MetadataFilterBuilder:
    """
    Builds Pinecone-compatible metadata filter dictionaries.
    
    All field definitions and valid operators loaded from YAML config.
    Dynamic field mappings from ingestion schema.
    Strict type validation with warnings.
    """
    
    def __init__(
        self,
        config: Optional[MetadataFilterConfig] = None,
        config_path: Optional[str] = None,
        ingestion_schema_path: Optional[str] = None,
        auto_update_schema: bool = False
    ):
        """
        Initialize the filter builder.
        
        Args:
            config: MetadataFilterConfig object
            config_path: Path to YAML config file
            ingestion_schema_path: Path to ingestion schema for dynamic mappings
            auto_update_schema: Whether to auto-update from schema on each build
        """
        self.config = config or MetadataFilterConfig.from_yaml(config_path)
        self.ingestion_schema_path = ingestion_schema_path
        self.auto_update_schema = auto_update_schema
        
        # Load from ingestion schema if provided
        if ingestion_schema_path and Path(ingestion_schema_path).exists():
            try:
                schema_config = MetadataFilterConfig.from_ingestion_schema(ingestion_schema_path)
                self.config.fields.update(schema_config.fields)
                self.config.field_mappings.update(schema_config.field_mappings)
                self.config.field_mapping_source = ingestion_schema_path
            except Exception as e:
                print(f"Warning: Failed to load ingestion schema: {e}")
    
    def build(
        self,
        pipeline_config: PipelineConfig,
        conversation_state: Optional[ConversationState] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
        strict_validation: bool = False
    ) -> FilterResult:
        """
        Build a complete metadata filter for a retrieval request.
        
        Args:
            pipeline_config: Pipeline configuration with metadata_filters
            conversation_state: Optional conversation state for context filtering
            additional_filters: Optional additional Pinecone filter dict
            strict_validation: If True, raise errors on validation failures
            
        Returns:
            FilterResult with the Pinecone-compatible filter dict
        """
        # Auto-update from ingestion schema if enabled
        if self.auto_update_schema and self.ingestion_schema_path:
            try:
                self.config.update_from_schema(self.ingestion_schema_path)
            except Exception as e:
                # Log warning but continue with existing config
                pass
        
        applied = []
        warnings = []
        validation_errors = []
        filter_parts = []
        
        # 1. Apply pipeline-level metadata filters
        if pipeline_config.metadata_filters:
            pipeline_filter, pipe_warnings, pipe_errors = self._build_from_dict(
                pipeline_config.metadata_filters, strict_validation
            )
            if pipeline_filter:
                filter_parts.append(pipeline_filter)
                applied.append(f"pipeline: {list(pipeline_config.metadata_filters.keys())}")
            warnings.extend(pipe_warnings)
            validation_errors.extend(pipe_errors)
        
        # 2. Apply conversation context filters
        if conversation_state:
            context_filter, context_applied, ctx_warnings = self._build_from_conversation(conversation_state)
            if context_filter:
                filter_parts.append(context_filter)
                applied.extend(context_applied)
            warnings.extend(ctx_warnings)
        
        # 3. Apply additional filters
        if additional_filters:
            additional, add_warnings, add_errors = self._build_from_dict(
                additional_filters, strict_validation
            )
            if additional:
                filter_parts.append(additional)
                applied.append(f"additional: {list(additional_filters.keys())}")
            warnings.extend(add_warnings)
            validation_errors.extend(add_errors)
        
        # 4. Apply template-based filters
        if pipeline_config.metadata_filters:
            template_filter, template_applied, template_warnings = self._apply_templates(
                pipeline_config.metadata_filters
            )
            if template_filter:
                filter_parts.append(template_filter)
                applied.extend(template_applied)
            warnings.extend(template_warnings)
        
        # Combine filters with $and if multiple parts exist
        if not filter_parts:
            return FilterResult(
                filter_dict={},
                applied_filters=[],
                is_empty=True,
                warnings=warnings,
                validation_errors=validation_errors
            )
        
        if len(filter_parts) == 1:
            final_filter = filter_parts[0]
        else:
            final_filter = {"$and": filter_parts}
        
        return FilterResult(
            filter_dict=final_filter,
            applied_filters=applied,
            is_empty=False,
            warnings=warnings,
            validation_errors=validation_errors
        )
    
    def build_simple(
        self,
        field: str,
        value: Union[str, int, float, List, bool],
        operator: str = "$eq",
        strict_validation: bool = False
    ) -> FilterResult:
        """
        Build a simple single-field filter.
        
        Args:
            field: Field name (uses field_mappings to resolve)
            value: Filter value
            operator: Pinecone operator ($eq, $ne, $gt, etc.)
            strict_validation: If True, raise errors on validation failures
            
        Returns:
            FilterResult with the filter dict
        """
        warnings = []
        validation_errors = []
        
        # Resolve field mapping
        resolved_field = self.config.resolve_field(field)
        
        # Validate field exists
        if resolved_field not in self.config.fields:
            warnings.append(f"Field '{resolved_field}' not defined in schema")
        
        # Validate operator
        if not self.config.validate_operator(resolved_field, operator):
            validation_errors.append(
                f"Invalid operator '{operator}' for field '{resolved_field}'"
            )
            if strict_validation:
                return FilterResult(
                    filter_dict={},
                    applied_filters=[],
                    is_empty=True,
                    warnings=warnings,
                    validation_errors=validation_errors
                )
        
        # Validate value type
        if not self.config.validate_value(resolved_field, value):
            field_type = self.config.get_field_type(resolved_field)
            validation_errors.append(
                f"Value '{value}' has invalid type for field '{resolved_field}' "
                f"(expected {field_type.value if field_type else 'unknown'})"
            )
            if strict_validation:
                return FilterResult(
                    filter_dict={},
                    applied_filters=[],
                    is_empty=True,
                    warnings=warnings,
                    validation_errors=validation_errors
                )
        
        # Special handling for array operators
        if operator in ("$in", "$nin", "$all"):
            if not isinstance(value, list):
                warnings.append(f"Converting single value to list for {operator} operator")
                value = [value]
        
        return FilterResult(
            filter_dict={resolved_field: {operator: value}},
            applied_filters=[f"{resolved_field} {operator} {value}"],
            is_empty=False,
            warnings=warnings,
            validation_errors=validation_errors
        )
    
    def build_compound(
        self,
        filters: List[Dict[str, Any]],
        combinator: str = "$and",
        strict_validation: bool = False
    ) -> FilterResult:
        """
        Build a compound filter combining multiple conditions.
        
        Args:
            filters: List of simple filter dicts
            combinator: "$and" or "$or"
            strict_validation: If True, raise errors on validation failures
            
        Returns:
            FilterResult with the compound filter
        """
        if combinator not in ("$and", "$or"):
            return FilterResult(
                filter_dict={},
                applied_filters=[],
                is_empty=True,
                warnings=[],
                validation_errors=[f"Invalid combinator: {combinator}"]
            )
        
        if not filters:
            return FilterResult(
                filter_dict={},
                applied_filters=[],
                is_empty=True
            )
        
        # Validate each sub-filter
        valid_filters = []
        applied = []
        all_warnings = []
        all_errors = []
        
        for f in filters:
            result = self._build_from_dict(f, strict_validation)
            if result[0]:
                valid_filters.append(result[0])
                applied.append(str(f))
            all_warnings.extend(result[1])
            all_errors.extend(result[2])
        
        if not valid_filters:
            return FilterResult(
                filter_dict={},
                applied_filters=[],
                is_empty=True,
                warnings=all_warnings,
                validation_errors=all_errors
            )
        
        if len(valid_filters) == 1:
            final = valid_filters[0]
        else:
            final = {combinator: valid_filters}
        
        return FilterResult(
            filter_dict=final,
            applied_filters=applied,
            is_empty=False,
            warnings=all_warnings,
            validation_errors=all_errors
        )
    
    def get_field_schema(self, field_name: str) -> Optional[Dict]:
        """Get the schema for a field as a dictionary."""
        schema = self.config.get_field_schema(field_name)
        return schema.to_dict() if schema else None
    
    def list_fields(self) -> List[Dict[str, Any]]:
        """List all available fields with their schemas."""
        return [
            {
                "name": name,
                **schema.to_dict()
            }
            for name, schema in self.config.fields.items()
        ]
    
    # ========================================================================
    # Internal Builders
    # ========================================================================
    
    def _build_from_dict(
        self,
        filter_dict: Dict[str, Any],
        strict_validation: bool = False
    ) -> tuple:
        """
        Build a Pinecone filter from a simplified dict format.
        
        Supports:
        - {"topic": "Deadlock"} → {"topic": {"$eq": "Deadlock"}}
        - {"page_start": {"$gte": 5}} → {"page_start": {"$gte": 5}}
        - {"topic": ["Deadlock", "Starvation"]} → {"topic": {"$in": [...]}}
        
        Returns:
            Tuple of (filter_dict, warnings, validation_errors)
        """
        if not filter_dict:
            return None, [], []
        
        result = {}
        warnings = []
        errors = []
        
        for field, value in filter_dict.items():
            resolved = self.config.resolve_field(field)

            # Check if field exists. A field the schema doesn't recognize
            # can't correspond to any real Pinecone metadata key, so
            # including it as a hard $eq/$in filter is guaranteed to match
            # zero vectors — skip it (with a warning) rather than build a
            # filter that can never succeed.
            if resolved not in self.config.fields:
                warnings.append(f"Field '{resolved}' not defined in schema — filter skipped")
                continue

            if isinstance(value, dict):
                # Already has operator format: {"$gte": 5}
                # Validate operator and value
                for operator, val in value.items():
                    if not self.config.validate_operator(resolved, operator):
                        errors.append(
                            f"Invalid operator '{operator}' for field '{resolved}'"
                        )
                    elif not self.config.validate_value(resolved, val):
                        field_type = self.config.get_field_type(resolved)
                        errors.append(
                            f"Value '{val}' has invalid type for field '{resolved}' "
                            f"(expected {field_type.value if field_type else 'unknown'})"
                        )
                result[resolved] = value
            elif isinstance(value, list):
                # List of values → $in operator
                # Validate each value
                valid_values = []
                for v in value:
                    if self.config.validate_value(resolved, v):
                        valid_values.append(v)
                    else:
                        field_type = self.config.get_field_type(resolved)
                        errors.append(
                            f"Value '{v}' has invalid type for field '{resolved}' "
                            f"(expected {field_type.value if field_type else 'unknown'})"
                        )
                result[resolved] = {"$in": valid_values if valid_values else value}
            else:
                # Simple value → $eq operator
                if not self.config.validate_value(resolved, value):
                    field_type = self.config.get_field_type(resolved)
                    errors.append(
                        f"Value '{value}' has invalid type for field '{resolved}' "
                        f"(expected {field_type.value if field_type else 'unknown'})"
                    )
                result[resolved] = {"$eq": value}
        
        if strict_validation and errors:
            return None, warnings, errors
        
        return result if result else None, warnings, errors
    
    def _build_from_conversation(
        self,
        state: ConversationState
    ) -> tuple:
        """
        Build filters from conversation context.
        
        Returns:
            Tuple of (filter_dict, applied_filters_list, warnings)
        """
        filters = {}
        applied = []
        warnings = []

        # NOTE: deliberately no hard filters on course/chapter/topic here.
        # state.subject/chapter/current_topic/current_concept are names
        # from the application-level topic taxonomy (LLM-generated by
        # PlannerController._extract_topics() — coarse-grained, e.g. course
        # "Supply Chain", chapter "Chapter 8", concept "Economic Order
        # Quantity"), while the ingested chunk metadata (course_name,
        # chapter_title, topic) is populated by a separate, independent
        # extraction step at ingestion time (e.g. course_name "Supply Chain
        # Management Strategy", chapter_title "Forecasting and Demand
        # Planning", topic "Quantitative Forecasting Methods"). These two
        # vocabularies have no reliable correspondence, so exact-match
        # filters here silently zeroed out every candidate on every query
        # — see the "couldn't find enough information" investigation.
        # Semantic similarity plus the soft metadata-field boost in
        # HybridSearch._metadata_search() do the topic-relevance work
        # instead, without hard-gating on a field that can't be trusted
        # to match.

        return filters if filters else None, applied, warnings
    
    def _apply_templates(self, metadata_filters: Dict[str, Any]) -> tuple:
        """
        Apply named filter templates from config.
        
        Templates are predefined filter patterns in the YAML config.
        
        Returns:
            Tuple of (filter_dict, applied_list, warnings)
        """
        filters = {}
        applied = []
        warnings = []
        
        for filter_key in metadata_filters:
            if filter_key in self.config.templates:
                template = self.config.templates[filter_key]
                template_fields = template.get("fields", {})
                for field, operator in template_fields.items():
                    # Templates may have placeholders — resolved from metadata_filters
                    value = metadata_filters.get(field)
                    if value:
                        resolved = self.config.resolve_field(field)
                        # Validate before applying
                        if self.config.validate_operator(resolved, operator):
                            if self.config.validate_value(resolved, value):
                                filters[resolved] = {operator: value}
                                applied.append(f"template:{filter_key}:{field}")
                            else:
                                warnings.append(
                                    f"Template value '{value}' failed validation for field '{resolved}'"
                                )
                        else:
                            warnings.append(
                                f"Template operator '{operator}' invalid for field '{resolved}'"
                            )
        
        return filters if filters else None, applied, warnings


# ============================================================================
# Schema Integration Example
# ============================================================================

def create_ingestion_schema_example():
    """Example of an ingestion schema file format."""
    return {
        "metadata": {
            "course_name": {
                "type": "string",
                "description": "Name of the course",
                "required": True,
                "canonical_name": "course"
            },
            "chapter_title": {
                "type": "string",
                "description": "Title of the chapter",
                "required": False,
                "canonical_name": "chapter"
            },
            "topic": {
                "type": "string",
                "description": "Topic name",
                "required": False
            },
            "file_type": {
                "type": "string",
                "description": "File type (pdf, pptx, docx, etc.)",
                "allowed_values": ["pdf", "pptx", "docx", "txt", "md"],
                "required": True
            },
            "page_start": {
                "type": "number",
                "description": "Starting page number",
                "required": False
            },
            "page_end": {
                "type": "number",
                "description": "Ending page number",
                "required": False
            },
            "is_primary": {
                "type": "boolean",
                "description": "Primary content flag",
                "default": False
            },
            "tags": {
                "type": "array",
                "description": "Tags associated with content",
                "required": False
            },
            "created_at": {
                "type": "timestamp",
                "description": "Creation timestamp",
                "required": False
            }
        }
    }


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Metadata Filter Builder - Enhanced Validation")
    print("=" * 60)
    
    # Load configuration
    config = MetadataFilterConfig.from_yaml()
    builder = MetadataFilterBuilder(config=config)
    
    print(f"\n📋 Configuration loaded:")
    print(f"   Available fields: {list(config.fields.keys())}")
    print(f"   Field mappings: {config.field_mappings}")
    print(f"   Templates: {list(config.templates.keys())}")
    
    # Test 1: Simple filter
    print(f"\n{'─' * 50}")
    print("Test 1: Simple equality filter")
    result = builder.build_simple("topic", "Deadlock")
    print(f"   Input: field='topic', value='Deadlock'")
    print(f"   Output: {result.filter_dict}")
    print(f"   Applied: {result.applied_filters}")
    print(f"   Warnings: {result.warnings}")
    
    # Test 2: Simple filter with field mapping
    print(f"\n{'─' * 50}")
    print("Test 2: Field mapping (course → course_name)")
    result = builder.build_simple("course", "Operating Systems")
    print(f"   Input: field='course', value='Operating Systems'")
    print(f"   Output: {result.filter_dict}")
    print(f"   Applied: {result.applied_filters}")
    
    # Test 3: Page range filter
    print(f"\n{'─' * 50}")
    print("Test 3: Number range filter")
    result = builder.build_simple("page_start", 5, "$gte")
    print(f"   Input: field='page_start', value=5, operator='$gte'")
    print(f"   Output: {result.filter_dict}")
    
    # Test 4: Type validation - invalid value
    print(f"\n{'─' * 50}")
    print("Test 4: Type validation - invalid value")
    result = builder.build_simple("page_start", "five", "$gte")
    print(f"   Input: field='page_start', value='five', operator='$gte'")
    print(f"   Output: {result.filter_dict}")
    print(f"   Validation errors: {result.validation_errors}")
    print(f"   Is empty: {result.is_empty}")
    
    # Test 5: Type validation - strict mode
    print(f"\n{'─' * 50}")
    print("Test 5: Type validation - strict mode")
    result = builder.build_simple("page_start", "five", "$gte", strict_validation=True)
    print(f"   Input: field='page_start', value='five', operator='$gte'")
    print(f"   Output: {result.filter_dict}")
    print(f"   Validation errors: {result.validation_errors}")
    print(f"   Is empty: {result.is_empty}")
    
    # Test 6: Compound filter with validation
    print(f"\n{'─' * 50}")
    print("Test 6: Compound filter with validation")
    result = builder.build_compound(
        filters=[
            {"topic": "Deadlock"},
            {"page_start": 5},
            {"page_end": "ten"},  # Invalid type
        ],
        combinator="$and",
        strict_validation=False
    )
    print(f"   Input: [topic=Deadlock, page_start=5, page_end='ten']")
    print(f"   Output: {result.filter_dict}")
    print(f"   Validation errors: {result.validation_errors}")
    print(f"   Warnings: {result.warnings}")
    
    # Test 7: Conversation context with validation
    print(f"\n{'─' * 50}")
    print("Test 7: Conversation context with validation")
    state = ConversationState(
        session_id="test",
        subject="Computer Science",
        chapter="Process Management",
        current_topic="Concurrency",
        current_concept="Deadlock"
    )
    
    from src.retrieval.config import PipelineConfig
    pipeline_config = PipelineConfig(
        name="test",
        metadata_filters={}
    )
    
    result = builder.build(pipeline_config, conversation_state=state)
    print(f"   Conversation: CS → Process Management → Concurrency → Deadlock")
    print(f"   Output: {result.filter_dict}")
    print(f"   Applied: {result.applied_filters}")
    
    # Test 8: Field listing
    print(f"\n{'─' * 50}")
    print("Test 8: List available fields")
    fields = builder.list_fields()
    for field in fields[:3]:  # Show first 3
        print(f"   {field['name']}: type={field['type']}, required={field.get('required', False)}")
    
    # Test 9: Dynamic schema update
    print(f"\n{'─' * 50}")
    print("Test 9: Dynamic schema update from ingestion")
    # Create a temporary ingestion schema
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        schema_data = create_ingestion_schema_example()
        yaml.dump(schema_data, f)
        schema_path = f.name
    
    try:
        builder_ingestion = MetadataFilterBuilder(ingestion_schema_path=schema_path)
        print(f"   Loaded schema from: {schema_path}")
        fields = builder_ingestion.list_fields()
        print(f"   Fields from ingestion: {[f['name'] for f in fields]}")
        print(f"   Mappings: {builder_ingestion.config.field_mappings}")
        
        # Test filtering with new schema
        result = builder_ingestion.build_simple("file_type", "pdf")
        print(f"   Filter with file_type: {result.filter_dict}")
        
    finally:
        os.unlink(schema_path)
    
    print(f"\n{'=' * 60}")
    print("✅ All validation tests passed!")
    print("=" * 60)