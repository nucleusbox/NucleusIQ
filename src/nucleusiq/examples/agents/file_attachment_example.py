"""
Example: File Attachments with Agents (Framework-Level)

NucleusIQ provides a unified Attachment API that works with ANY LLM
provider.  You write the same code regardless of whether you're using
MockLLM, OpenAI, Anthropic, or a custom provider.

Attachment Types (all providers):
    TEXT         - plain text content (.txt, .md, .csv, .json, etc.)
    PDF          - PDF document (bytes or file content)
    FILE_BYTES   - raw bytes (any file format)
    FILE_BASE64  - pre-encoded base64 file data (from APIs, no double-encoding)
    IMAGE_URL    - remote image URL (vision models)
    IMAGE_BASE64 - local image as base64 (vision models)
    FILE_URL     - remote file URL (provider fetches server-side)

How it works:
    - Framework-level: Extracts text from files client-side (works everywhere)
    - Provider-native:  If your LLM provider supports native file input
      (e.g. OpenAI), the provider's process_attachments() sends raw files
      for server-side processing automatically.  You don't change your code.

Provider Capability Introspection:
    agent.llm.NATIVE_ATTACHMENT_TYPES       # what types are optimised
    agent.llm.SUPPORTED_FILE_EXTENSIONS     # what file extensions work natively
    agent.llm.describe_attachment_support()  # human-readable summary

Scenarios:
    1. Text file attachment
    2. Raw file bytes (FILE_BYTES)
    3. Pre-encoded base64 file (FILE_BASE64)
    4. Image URL (vision)
    5. Image base64 (local image)
    6. PDF document
    7. FILE_URL (provider-native remote file)
    8. Multiple mixed attachments
    9. Streaming with attachments
   10. Size validation
   11. Provider capability introspection

Run with:
    python examples/agents/file_attachment_example.py
"""

import asyncio
import base64
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent, Attachment, AttachmentType, Task
from nucleusiq.agents.attachments import MAX_FILE_SIZE_BYTES, AttachmentProcessor
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.llms.mock_llm import MockLLM

logging.basicConfig(level=logging.WARNING)


def make_agent(name: str, role: str) -> Agent:
    return Agent(
        name=name,
        role=role,
        objective=f"{role} tasks",
        llm=MockLLM(),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )


# ======================================================================
# 1. TEXT - plain text content
# ======================================================================


async def example_text():
    """AttachmentType.TEXT - for plain text content (.txt, .md, .csv, etc.)."""
    print("\n" + "=" * 70)
    print("1. TEXT Attachment")
    print("=" * 70)

    agent = make_agent("DocBot", "Document analyst")

    task = Task(
        id="ex-text",
        objective="Summarize the key metrics from this report.",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT,
                data="Revenue: $2.5M (up 30%)\nActive users: 15,000\nChurn: 2%",
                name="q4_report.txt",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  Name:     {task.attachments[0].name}")
    print(f"  Data:     (plain string, {len(task.attachments[0].data)} chars)")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 2. FILE_BYTES - raw file bytes
# ======================================================================


async def example_file_bytes():
    """AttachmentType.FILE_BYTES - for raw bytes from open('file', 'rb').read()."""
    print("\n" + "=" * 70)
    print("2. FILE_BYTES Attachment")
    print("=" * 70)

    agent = make_agent("DataBot", "Data analyst")

    csv_bytes = b"product,revenue,growth\nWidget A,195000,30%\nWidget B,220000,10%"

    task = Task(
        id="ex-bytes",
        objective="Which product grew faster?",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_BYTES,
                data=csv_bytes,
                name="products.csv",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  Name:     {task.attachments[0].name}")
    print(f"  Data:     (raw bytes, {len(csv_bytes)} bytes)")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 3. FILE_BASE64 - pre-encoded base64 file data
# ======================================================================


async def example_file_base64():
    """AttachmentType.FILE_BASE64 - for data already in base64 (from APIs)."""
    print("\n" + "=" * 70)
    print("3. FILE_BASE64 Attachment")
    print("=" * 70)

    agent = make_agent("APIBot", "API data processor")

    original_content = b"id,name,score\n1,Alice,95\n2,Bob,87\n3,Carol,92"
    b64_data = base64.b64encode(original_content).decode()

    task = Task(
        id="ex-b64",
        objective="What is the average score?",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_BASE64,
                data=b64_data,
                name="scores.csv",
                mime_type="text/csv",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  Name:     {task.attachments[0].name}")
    print(f"  Data:     (pre-encoded base64, {len(b64_data)} chars)")
    print("  Note:     Data is NOT re-encoded; passed directly to provider")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 4. IMAGE_URL - remote image for vision
# ======================================================================


async def example_image_url():
    """AttachmentType.IMAGE_URL - for remote image URLs (vision models)."""
    print("\n" + "=" * 70)
    print("4. IMAGE_URL Attachment")
    print("=" * 70)

    agent = make_agent("VisionBot", "Image analyst")

    task = Task(
        id="ex-imgurl",
        objective="Describe the chart and identify any trends.",
        attachments=[
            Attachment(
                type=AttachmentType.IMAGE_URL,
                data="https://example.com/charts/revenue_2025.png",
                detail="high",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  URL:      {task.attachments[0].data}")
    print(f"  Detail:   {task.attachments[0].detail}")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 5. IMAGE_BASE64 - local image as base64
# ======================================================================


async def example_image_base64():
    """AttachmentType.IMAGE_BASE64 - for local images as bytes or base64."""
    print("\n" + "=" * 70)
    print("5. IMAGE_BASE64 Attachment")
    print("=" * 70)

    agent = make_agent("PhotoBot", "Photo analyst")

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    task = Task(
        id="ex-imgb64",
        objective="What is shown in this image?",
        attachments=[
            Attachment(
                type=AttachmentType.IMAGE_BASE64,
                data=fake_png,
                mime_type="image/png",
                detail="auto",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  MIME:     {task.attachments[0].mime_type}")
    print(f"  Data:     (raw image bytes, {len(fake_png)} bytes)")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 6. PDF - document bytes
# ======================================================================


async def example_pdf():
    """AttachmentType.PDF - for PDF documents."""
    print("\n" + "=" * 70)
    print("6. PDF Attachment")
    print("=" * 70)

    agent = make_agent("PDFBot", "Document analyst")

    pdf_bytes = b"%PDF-1.4 sample quarterly report"

    task = Task(
        id="ex-pdf",
        objective="Extract the key findings from this report.",
        attachments=[
            Attachment(
                type=AttachmentType.PDF,
                data=pdf_bytes,
                name="Q4_report.pdf",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  Name:     {task.attachments[0].name}")
    print(f"  Data:     (PDF bytes, {len(pdf_bytes)} bytes)")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 7. FILE_URL - remote file reference
# ======================================================================


async def example_file_url():
    """AttachmentType.FILE_URL - for remote file references."""
    print("\n" + "=" * 70)
    print("7. FILE_URL Attachment")
    print("=" * 70)

    agent = make_agent("RemoteBot", "Document processor")

    task = Task(
        id="ex-url",
        objective="Extract all tables from this PDF.",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_URL,
                data="https://storage.example.com/reports/annual_2025.pdf",
                name="annual_2025.pdf",
            ),
        ],
    )

    print(f"  Type:     {task.attachments[0].type.value}")
    print(f"  URL:      {task.attachments[0].data}")
    print("  Note:     Provider handles retrieval (or fallback text note)")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 8. Multiple mixed attachments
# ======================================================================


async def example_mixed():
    """Combine multiple attachment types in a single task."""
    print("\n" + "=" * 70)
    print("8. Mixed Attachments (text + image + data)")
    print("=" * 70)

    agent = make_agent("ResearchBot", "Research analyst")

    task = Task(
        id="ex-mixed",
        objective="Compare the report text with the chart and raw data.",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT,
                data="Revenue grew 30% in Q4. Churn dropped to 2%.",
                name="summary.txt",
            ),
            Attachment(
                type=AttachmentType.IMAGE_URL,
                data="https://example.com/charts/q4_chart.png",
                detail="high",
            ),
            Attachment(
                type=AttachmentType.FILE_BYTES,
                data=b"quarter,revenue\nQ1,1.8M\nQ2,1.9M\nQ3,2.1M\nQ4,2.5M",
                name="revenue.csv",
            ),
        ],
    )

    print("  Attachments:")
    for att in task.attachments:
        label = att.name or str(att.data)[:40]
        print(f"    - {att.type.value:15s}  {label}")

    result = await agent.execute(task)
    print(f"  Result:   {str(result)[:100]}...")


# ======================================================================
# 9. Streaming with attachments
# ======================================================================


async def example_streaming():
    """Streaming works identically with or without attachments."""
    print("\n" + "=" * 70)
    print("9. Streaming with Attachments")
    print("=" * 70)

    agent = Agent(
        name="StreamBot",
        role="Writer",
        objective="Write summaries",
        llm=MockLLM(stream_chunk_size=8),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(
        id="ex-stream",
        objective="Summarize the meeting notes.",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT,
                data="Meeting: Jan 15 launch confirmed. Design approved.",
                name="notes.txt",
            ),
        ],
    )

    print("  Streaming: ", end="")
    async for event in agent.execute_stream(task):
        if event.type == "token":
            print(event.token, end="", flush=True)
        elif event.type == "complete":
            print("\n  Done.")


# ======================================================================
# 10. Size validation
# ======================================================================


async def example_size_validation():
    """File size is validated automatically (50 MB limit)."""
    print("\n" + "=" * 70)
    print("10. File Size Validation")
    print("=" * 70)

    small = Attachment(type=AttachmentType.TEXT, data="Hello", name="s.txt")
    try:
        AttachmentProcessor.validate_size(small)
        print(f"  small.txt ({len(small.data)} B):  PASS")
    except ValueError as e:
        print(f"  small.txt: FAIL - {e}")

    print(f"  Max allowed: {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB")
    print("  Enforced by: AttachmentProcessor.validate_size()")
    print("               and provider's process_attachments()")


# ======================================================================
# 11. Provider capability introspection
# ======================================================================


async def example_introspection():
    """
    Every LLM declares what attachment types it handles natively.

    Use this to understand how your chosen provider processes files:
        agent.llm.NATIVE_ATTACHMENT_TYPES        - types processed server-side
        agent.llm.SUPPORTED_FILE_EXTENSIONS      - file extensions for native routing
        agent.llm.describe_attachment_support()   - full structured summary

    This mirrors the NATIVE_TOOL_TYPES pattern for native tools.
    """
    print("\n" + "=" * 70)
    print("11. Provider Capability Introspection")
    print("=" * 70)

    llm = MockLLM()

    print(f"\n  Provider: {type(llm).__name__}")
    print(
        f"  NATIVE_ATTACHMENT_TYPES:    {sorted(t.value for t in llm.NATIVE_ATTACHMENT_TYPES) or '(none)'}"
    )
    print(
        f"  SUPPORTED_FILE_EXTENSIONS:  {sorted(llm.SUPPORTED_FILE_EXTENSIONS) or '(none)'}"
    )

    info = llm.describe_attachment_support()
    print("\n  describe_attachment_support():")
    for atype, desc in info["type_details"].items():
        print(f"    {atype:15s} -> {desc}")

    print(f"\n  Notes: {info['notes']}")

    print("\n  How to use with OpenAI (when installed):")
    print("    from nucleusiq_openai import BaseOpenAI")
    print("    llm = BaseOpenAI(model_name='gpt-4o')")
    print("    llm.NATIVE_ATTACHMENT_TYPES  # -> {pdf, file_base64, file_url, ...}")
    print("    llm.SUPPORTED_FILE_EXTENSIONS  # -> {.pdf, .csv, .xlsx, ...}")
    print("    llm.describe_attachment_support()  # -> full details")


# ======================================================================
# Main
# ======================================================================


async def main():
    print("NucleusIQ File Attachment Guide")
    print("=" * 70)
    print()
    print("The Attachment API is provider-agnostic.  The same code works with")
    print("MockLLM, OpenAI, Anthropic, or any custom provider.")
    print()
    print("Attachment Types:")
    for t in sorted(AttachmentType, key=lambda x: x.value):
        print(f"  {t.value:15s} ({t.name})")
    print()
    print("Provider capability (mirrors NATIVE_TOOL_TYPES pattern):")
    print("  llm.NATIVE_ATTACHMENT_TYPES       -> what's optimised")
    print("  llm.SUPPORTED_FILE_EXTENSIONS     -> what extensions work natively")
    print("  llm.describe_attachment_support()  -> full summary")

    await example_text()
    await example_file_bytes()
    await example_file_base64()
    await example_image_url()
    await example_image_base64()
    await example_pdf()
    await example_file_url()
    await example_mixed()
    await example_streaming()
    await example_size_validation()
    await example_introspection()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
