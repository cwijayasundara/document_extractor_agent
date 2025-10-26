"""Main entry point for the iterative extraction workflow.

This module handles:
- Workflow initialization and execution
- User interaction for schema approval
- Progress event display
- Result output
"""

import asyncio
from pathlib import Path

# Import workflow components from package
from .extraction_workflow import (
    IterativeExtractionWorkflow,
    ProgressEvent,
    ProposedSchema,
    ApprovedSchema,
)

async def main():
    """Execute the iterative extraction workflow.

    This function:
    1. Initializes the workflow
    2. Starts the workflow with a file path and prompt
    3. Handles progress events
    4. Manages the schema approval interaction loop
    5. Outputs the final extracted data and agent ID
    """
    # Initialize workflow with no timeout
    wf = IterativeExtractionWorkflow(timeout=None)

    # Get the project root directory (parent of research_2)
    project_root = Path(__file__).parent.parent
    file_path = project_root / "docs" / "IMG_4693.JPEG"

    # Start the workflow
    handler = wf.run(
        file_path=str(file_path),
        prompt="Extract all the important information from the invoice.",
    )

    # Stream events and handle user interaction
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

    # Wait for final result
    result = await handler
    print(f"Agent ID: {result.agent_id}", flush=True)
    print(f"Extracted data: {result.data}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
