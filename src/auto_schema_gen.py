import warnings
warnings.filterwarnings("ignore")
from openai import AsyncOpenAI
from llama_cloud_services.extract import (
    LlamaExtract,
    ExtractConfig,
    ExtractMode,
)
from llama_cloud_services.parse import LlamaParse
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
)
import json
import re
import uuid
import asyncio
from jsonschema import Draft202012Validator
from workflows import Context, Workflow, step
from workflows.resource import Resource
from typing import Annotated

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-nano"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_CLOUD_PROJECT_ID")
LLAMA_CLOUD_ORG_ID = os.getenv("LLAMA_CLOUD_ORG_ID")

async def get_parse_client(**kwargs):
    return LlamaParse(api_key=LLAMA_CLOUD_API_KEY, 
        parse_mode="parse_page_with_agent",
        model="gemini-2.5-flash",
        high_res_ocr=True,
        adaptive_long_table=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=False,
        take_screenshot=False)

async def get_extract_client(**kwargs):
    return LlamaExtract(api_key=LLAMA_CLOUD_API_KEY, name="extraction_agent_v1")

async def get_openai_client(**kwargs):
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

class WorkflowState(BaseModel):
    file_path: str | None = None
    file_content: str | None = None
    current_schema: dict | None = None
    current_feedback: str | None = None
    original_prompt: str | None = None

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

class InputEvent(StartEvent):
    """Start the workflow by providing a file path and a prompt."""
    file_path: str
    prompt: str

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

class ExtractedData(StopEvent):
    """Outputs the extracted data and the agent ID from the workflow run."""
    data: dict
    agent_id: str

class ProgressEvent(Event):
    """Propagates a progress message to the user as the workflow runs."""
    msg: str



class IterativeExtractionWorkflow(Workflow):
    @step
    async def parse_file(
        self,
        ev: InputEvent,
        ctx: Context[WorkflowState],
        parser: Annotated[LlamaParse, Resource(get_parse_client)],
    ) -> ParsedContent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"Parsing file: {ev.file_path}"))
        result = await parser.aparse(ev.file_path)
        ctx.write_event_to_stream(ProgressEvent(msg="File parsed successfully"))

        # Update the state with the file content and path
        async with ctx.store.edit_state() as state:
            state.original_prompt = ev.prompt
            state.file_path = ev.file_path
            state.file_content = "\n\n".join([page.md for page in result.pages])

        return ParsedContent(file_content=state.file_content, prompt=ev.prompt)

    @step
    async def propose_schema(
        self,
        ev: ParsedContent,
        ctx: Context[WorkflowState],
        client: Annotated[AsyncOpenAI, Resource(get_openai_client)],
    ) -> ProposedSchema:
        ctx.write_event_to_stream(ProgressEvent(msg="Proposing schema"))

        # Inject feedback from previous attempts if available
        state = await ctx.store.get_state()
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
        async with ctx.store.edit_state() as state:
            state.current_feedback = ev.feedback

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

async def main():
    wf = IterativeExtractionWorkflow(timeout=None)

    handler = wf.run(
        file_path=os.path.abspath("docs/IMG_4693.JPEG"),
        prompt="Extract all the important information from the invoice.",
    )
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.msg, flush=True)
        elif isinstance(ev, ProposedSchema):
            print(f"Proposed schema: {ev.generated_schema}", flush=True)
            approved = input("Approve? (y/<reason>): ").strip().lower()
            print(f"Approved? {approved}", flush=True)
            if approved == "y":
                handler.ctx.send_event(ApprovedSchema(approved=True, feedback="Approved"))
            else:
                handler.ctx.send_event(ApprovedSchema(approved=False, feedback=approved))

    result = await handler
    print(f"Agent ID: {result.agent_id}", flush=True)
    print(f"Extracted data: {result.data}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
