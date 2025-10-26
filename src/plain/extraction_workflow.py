"""Extraction workflow module.

This module contains:
- Workflow state definition
- Event class definitions
- IterativeExtractionWorkflow implementation with all workflow steps
"""

from openai import AsyncOpenAI
from llama_cloud_services.extract import LlamaExtract, ExtractConfig, ExtractMode
from llama_cloud_services.parse import LlamaParse
from pydantic import BaseModel
from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows import Context, Workflow, step
from workflows.resource import Resource
from typing import Annotated
from jsonschema import Draft202012Validator
import json
import re
import uuid
import asyncio

from .config import (
    OPENAI_MODEL,
    get_parse_client,
    get_extract_client,
    get_openai_client,
)
from .cache_utils import (
    get_cached_parse,
    save_parsed_content,
    get_cache_path,
)
from .template_schema_generator import TemplateSchemaGenerator


# Workflow State Definition
class WorkflowState(BaseModel):
    """State model for the extraction workflow."""
    file_path: str | None = None
    file_content: str | None = None
    current_schema: dict | None = None
    current_feedback: str | None = None
    original_prompt: str | None = None
    template_path: str | None = None  # Optional Excel template path for manual mode


# Prompt Template
extract_prompt = """\
<file>
{file_content}
</file>

<user_prompt>
{prompt}
</user_prompt>
{past_attempt}

Given the file content above, and the user prompt, output a JSON schema that will be used to extract the data from the file.

Your JSON schema should have a root node with "type": "object" and fields inside "properties".

Wrap your schema in <schema>...</schema> tags.
"""


# Event Class Definitions
class InputEvent(StartEvent):
    """Start the workflow by providing a file path and a prompt."""
    file_path: str
    prompt: str
    template_path: str | None = None  # Optional Excel template for manual mode


class ParsedContent(Event):
    """Parses the content and carry forward the content and prompt."""
    file_content: str
    prompt: str


class RunExtraction(Event):
    """Runs the extraction on the provided file content and schema."""
    generated_schema: dict
    file_content: str


class ProposedSchema(InputRequiredEvent):
    """Proposes a schema for the extraction. Needs human approval."""
    generated_schema: dict


class ApprovedSchema(HumanResponseEvent):
    """Handles the human approval of the proposed schema."""
    approved: bool
    feedback: str
    edited_schema: dict | None = None  # Optional edited schema from UI


class ExtractedData(StopEvent):
    """Outputs the extracted data and the agent ID from the workflow run."""
    data: dict
    agent_id: str


class ProgressEvent(Event):
    """Propagates a progress message to the user as the workflow runs."""
    msg: str


# Workflow Implementation
class IterativeExtractionWorkflow(Workflow):
    """Iterative extraction workflow with schema approval loop.

    This workflow:
    1. Parses a document file
    2. Proposes a JSON schema for extraction
    3. Requests human approval (with iterative refinement based on feedback)
    4. Runs extraction using the approved schema
    """

    @step
    async def parse_file(
        self,
        ev: InputEvent,
        ctx: Context[WorkflowState],
        parser: Annotated[LlamaParse, Resource(get_parse_client)],
    ) -> ParsedContent:
        """Parse the input file using LlamaParse with caching.

        First checks if parsed content exists in cache. If found, loads from cache.
        Otherwise, parses the file using LlamaParse and saves to cache.

        Args:
            ev: Input event containing file path and prompt
            ctx: Workflow context for state management
            parser: LlamaParse client instance

        Returns:
            ParsedContent event with file content and prompt
        """
        ctx.write_event_to_stream(ProgressEvent(msg=f"Checking cache for: {ev.file_path}"))

        # Check cache first
        cached_content = get_cached_parse(ev.file_path)

        if cached_content is not None:
            # Cache hit - use cached content
            cache_path = get_cache_path(ev.file_path)
            ctx.write_event_to_stream(
                ProgressEvent(msg=f"✓ Found cached parse result: {cache_path}")
            )
            file_content = cached_content
        else:
            # Cache miss - parse the file
            ctx.write_event_to_stream(ProgressEvent(msg=f"Parsing file: {ev.file_path}"))
            result = await parser.aparse(ev.file_path)

            # Combine all pages into single content string
            file_content = "\n\n".join([page.md for page in result.pages])

            # Save to cache
            cache_path = save_parsed_content(ev.file_path, file_content)
            ctx.write_event_to_stream(
                ProgressEvent(msg=f"✓ Saved parse result to cache: {cache_path}")
            )

        # Update the state with the file content and path
        async with ctx.store.edit_state() as state:
            state.original_prompt = ev.prompt
            state.file_path = ev.file_path
            state.file_content = file_content
            state.template_path = ev.template_path  # Store template path for manual mode

        return ParsedContent(file_content=state.file_content, prompt=ev.prompt)

    @step
    async def propose_schema(
        self,
        ev: ParsedContent,
        ctx: Context[WorkflowState],
        client: Annotated[AsyncOpenAI, Resource(get_openai_client)],
    ) -> ProposedSchema:
        """Propose a JSON schema for extraction.

        Uses either template-based generation (Manual mode) or AI-based generation (Auto mode).
        If previous feedback exists, it incorporates that feedback to refine the schema.

        Args:
            ev: ParsedContent event with file content and prompt
            ctx: Workflow context for state management
            client: OpenAI client instance

        Returns:
            ProposedSchema event with generated schema

        Raises:
            Exception: If schema generation fails after 3 attempts
        """
        state = await ctx.store.get_state()

        # Check if template path is provided (Manual mode)
        if state.template_path:
            ctx.write_event_to_stream(ProgressEvent(msg="Generating schema from template"))

            try:
                # Generate schema from Excel template
                generator = TemplateSchemaGenerator(state.template_path)
                schema = generator.generate_schema(document_type="document")

                # Store schema in state
                async with ctx.store.edit_state() as state:
                    state.current_schema = schema

                # Get field summary for logging
                summary = generator.get_field_summary()
                ctx.write_event_to_stream(
                    ProgressEvent(
                        msg=f"✓ Schema generated from template: {summary['document_field_count']} document fields, {summary['line_item_field_count']} line item fields"
                    )
                )

                return ProposedSchema(generated_schema=schema)

            except Exception as e:
                ctx.write_event_to_stream(
                    ProgressEvent(msg=f"⚠ Template schema generation failed: {e}\n\nFalling back to AI-based generation")
                )
                # Fall through to AI-based generation below

        # Auto mode: Use AI-based schema generation
        ctx.write_event_to_stream(ProgressEvent(msg="Proposing schema using AI"))

        # Inject feedback from previous attempts if available
        if state.current_feedback and state.current_schema:
            past_attempt_str = f"\n<past_attempt>\n<feedback>{state.current_feedback}</feedback>\n<schema>{str(state.current_schema)}</schema>\n</past_attempt>\n"
        else:
            past_attempt_str = ""

        # Start the extraction process with a fresh chat history
        prompt = extract_prompt.format(
            file_content=ev.file_content,
            prompt=ev.prompt,
            past_attempt=past_attempt_str,
        )

        history = [{"role": "user", "content": prompt}]

        # Generate a new schema using OpenAI
        response = await client.chat.completions.create(
            messages=history,
            model=OPENAI_MODEL,
            temperature=1.0,
        )
        history.append({"role": "assistant", "content": response.choices[0].message.content})

        # Try to parse the schema from the response. If it fails, try again, using chat history
        # to keep track of failed attempts.
        attempts = 1
        schema = {}
        while attempts <= 3 and not schema:
            try:
                response_text = response.choices[0].message.content
                ctx.write_event_to_stream(
                    ProgressEvent(
                        msg=f"Attempting to parse schema string from:\n{response_text}"
                    )
                )
                json_str = re.sub(
                    r"<schema>([\s\S]*)<\/schema>", r"\1", response_text
                )

                # Validate the schema
                schema = json.loads(json_str)
                Draft202012Validator.check_schema(schema)

                async with ctx.store.edit_state() as state:
                    state.current_schema = schema

                break
            except Exception as e:
                ctx.write_event_to_stream(
                    ProgressEvent(msg=f"Schema parsing failed:\n{e}\n\nTrying again...")
                )
                history.append(
                    {"role": "user", "content": f"Error: {e}\n\nPlease try again."}
                )
                response = await client.chat.completions.create(
                    messages=history,
                    model=OPENAI_MODEL,
                    temperature=1.0,
                )
                history.append({"role": "assistant", "content": response.choices[0].message.content})
                attempts += 1

        if attempts > 3:
            raise Exception("Failed to propose a valid schema after 3 attempts!")

        ctx.write_event_to_stream(ProgressEvent(msg="Schema proposed successfully"))
        return ProposedSchema(generated_schema=schema)

    @step
    async def handle_schema_approval(
        self,
        ev: ApprovedSchema,
        ctx: Context[WorkflowState],
    ) -> ParsedContent | RunExtraction:
        """Handle the human approval or rejection of the proposed schema.

        Args:
            ev: ApprovedSchema event with approval status, feedback, and optional edited schema
            ctx: Workflow context for state management

        Returns:
            RunExtraction if approved, ParsedContent if rejected (to retry with feedback)
        """
        async with ctx.store.edit_state() as state:
            state.current_feedback = ev.feedback
            # If user provided an edited schema, use that instead of the current one
            if ev.edited_schema is not None:
                state.current_schema = ev.edited_schema

        # If the schema is approved, run the extraction. Otherwise, go back to the start and try again.
        if ev.approved:
            return RunExtraction(
                generated_schema=state.current_schema, file_content=state.file_content
            )
        else:
            return ParsedContent(
                file_content=state.file_content, prompt=state.original_prompt
            )

    @step
    async def run_extraction(
        self,
        ev: RunExtraction,
        ctx: Context[WorkflowState],
        extract: Annotated[LlamaExtract, Resource(get_extract_client)],
    ) -> ExtractedData:
        """Run the extraction using the approved schema.

        Args:
            ev: RunExtraction event with schema and file content
            ctx: Workflow context for state management
            extract: LlamaExtract client instance

        Returns:
            ExtractedData event with extracted data and agent ID

        Raises:
            Exception: If extraction fails
        """
        ctx.write_event_to_stream(ProgressEvent(msg="Running extraction"))

        # Persist an extraction agent + schema to llama-cloud
        agent = extract.create_agent(
            name=f"extraction_workflow_{uuid.uuid4()}",
            data_schema=ev.generated_schema,
            config=ExtractConfig(
                extraction_mode=ExtractMode.BALANCED,
            ),
        )

        try:
            # Run the extraction
            file_path = await ctx.store.get("file_path")
            result = await agent.aextract(files=file_path)

            if result.data is None:
                raise Exception(f"Extraction failed for file: {file_path}")

            return ExtractedData(data=result.data, agent_id=agent.id)
        finally:
            # Always cleanup the agent from Llama Cloud to prevent accumulation
            try:
                # Check if delete_agent method exists
                if hasattr(extract, 'delete_agent'):
                    delete_result = extract.delete_agent(agent.id)

                    # Handle both sync and async methods
                    if asyncio.iscoroutine(delete_result):
                        await delete_result

                    ctx.write_event_to_stream(
                        ProgressEvent(msg=f"✓ Cleaned up agent from Llama Cloud: {agent.id}")
                    )
                elif hasattr(agent, 'delete'):
                    # Try agent.delete() if extract.delete_agent doesn't exist
                    delete_result = agent.delete()
                    if asyncio.iscoroutine(delete_result):
                        await delete_result

                    ctx.write_event_to_stream(
                        ProgressEvent(msg=f"✓ Cleaned up agent from Llama Cloud: {agent.id}")
                    )
                else:
                    # No deletion method available
                    ctx.write_event_to_stream(
                        ProgressEvent(msg=f"ℹ️ Note: Agent {agent.id} cleanup not available (agent may persist in Llama Cloud)")
                    )
            except Exception as e:
                ctx.write_event_to_stream(
                    ProgressEvent(msg=f"⚠ Warning: Failed to delete agent {agent.id}: {e}")
                )
