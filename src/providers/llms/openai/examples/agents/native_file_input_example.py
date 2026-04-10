"""
Example: OpenAI Native File Input

Same Attachment API, smarter processing.

When you use the OpenAI provider, NucleusIQ automatically sends files
in OpenAI's native format for server-side processing.  This preserves
images in PDFs, augments spreadsheets, and avoids lossy client-side
text extraction.

You do NOT change your code.  The same Task + Attachment works with
any provider.  OpenAI just gets a better wire format under the hood.

Provider Capability (mirrors NATIVE_TOOL_TYPES pattern):
    llm.NATIVE_ATTACHMENT_TYPES       -> types processed server-side
    llm.SUPPORTED_FILE_EXTENSIONS     -> file extensions for native routing
    llm.describe_attachment_support() -> full structured summary

Scenarios:
    1. Provider capability introspection
    2. PDF native processing
    3. Structured data (CSV, JSON)
    4. FILE_BASE64 (pre-encoded, no double-encoding)
    5. FILE_URL (Responses API server-side fetch)
    6. Mixed files + images
    7. Streaming with file attachments

Requirements:
    - pip install nucleusiq-openai
    - Set OPENAI_API_KEY for real responses (runs with MockLLM otherwise)

Run with:
    python examples/agents/native_file_input_example.py
"""

import asyncio
import base64
import logging
import os
import sys

_src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, _src_dir)

from nucleusiq.agents import Agent, Attachment, AttachmentType, Task
from nucleusiq.agents.config import AgentConfig, ExecutionMode
from nucleusiq.llms.mock_llm import MockLLM
from nucleusiq.prompts.zero_shot import ZeroShotPrompt
from nucleusiq_openai import BaseOpenAI

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USE_REAL_LLM = os.getenv("OPENAI_API_KEY") is not None


def get_llm():
    if USE_REAL_LLM:
        return BaseOpenAI(model_name="gpt-4o", temperature=0.3)
    logger.info("Using MockLLM (set OPENAI_API_KEY for real OpenAI responses)")
    return MockLLM()


# ======================================================================
# 1. Provider capability introspection
# ======================================================================


async def example_capabilities():
    """
    Every LLM declares what attachment types it handles natively.

    This mirrors the NATIVE_TOOL_TYPES pattern for native tools.
    OpenAI declares 5 native types and 32 supported file extensions.
    """
    print("\n" + "=" * 70)
    print("1. Provider Capability Introspection")
    print("=" * 70)

    mock_llm = MockLLM()
    openai_llm = BaseOpenAI.__new__(BaseOpenAI)

    print("\n  --- MockLLM (generic provider) ---")
    print(
        f"  NATIVE_ATTACHMENT_TYPES:   {sorted(t.value for t in mock_llm.NATIVE_ATTACHMENT_TYPES) or '(none)'}"
    )
    print(
        f"  SUPPORTED_FILE_EXTENSIONS: {sorted(mock_llm.SUPPORTED_FILE_EXTENSIONS) or '(none)'}"
    )

    print("\n  --- BaseOpenAI (native file support) ---")
    print(
        f"  NATIVE_ATTACHMENT_TYPES:   {sorted(t.value for t in openai_llm.NATIVE_ATTACHMENT_TYPES)}"
    )
    print(
        f"  SUPPORTED_FILE_EXTENSIONS: {len(openai_llm.SUPPORTED_FILE_EXTENSIONS)} extensions"
    )
    print(f"    Sample: {sorted(list(openai_llm.SUPPORTED_FILE_EXTENSIONS))[:8]}...")

    print("\n  --- How each type is handled ---")
    info = get_llm().describe_attachment_support()
    for atype, desc in info["type_details"].items():
        print(f"    {atype:15s} -> {desc}")

    print(f"\n  Provider: {info['provider']}")
    print(f"  Notes:    {info['notes']}")


# ======================================================================
# 2. PDF - native server-side processing
# ======================================================================


async def example_pdf_native():
    """
    PDF sent as native file to OpenAI.

    OpenAI extracts text AND images from the PDF server-side.
    Framework-level would only extract text (via pdfplumber).
    """
    print("\n" + "=" * 70)
    print("2. PDF Native Processing")
    print("=" * 70)

    agent = Agent(
        name="PDFBot",
        role="Financial analyst",
        objective="Analyze documents",
        llm=get_llm(),
        prompt=ZeroShotPrompt().configure(
            system="You are a financial analyst. Summarize and extract key metrics from documents the user attaches.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    pdf_bytes = b"%PDF-1.4 quarterly report content"

    task = Task(
        id="pdf-native",
        objective="Summarize the key financial metrics from this report.",
        attachments=[
            Attachment(
                type=AttachmentType.PDF,
                data=pdf_bytes,
                name="Q4_report.pdf",
            ),
        ],
    )

    is_native = AttachmentType.PDF in agent.llm.NATIVE_ATTACHMENT_TYPES
    print(f"  Attachment: {task.attachments[0].name} ({len(pdf_bytes)} bytes)")
    print(f"  Native:     {is_native}")
    print(f"  Provider:   {type(agent.llm).__name__}")

    result = await agent.execute(task)
    print(f"  Result:     {str(result)[:100]}...")


# ======================================================================
# 3. Structured data (CSV, JSON)
# ======================================================================


async def example_structured_data():
    """
    Structured data files sent as native files to OpenAI.

    OpenAI augments CSV/XLSX/TSV with additional context.
    Framework-level treats them as plain text.
    """
    print("\n" + "=" * 70)
    print("3. Structured Data (CSV + JSON)")
    print("=" * 70)

    agent = Agent(
        name="DataBot",
        role="Data analyst",
        objective="Analyze data",
        llm=get_llm(),
        prompt=ZeroShotPrompt().configure(
            system="You are a data analyst. Answer questions using attached CSV, JSON, and structured files.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(
        id="data-native",
        objective="Which products met their Q4 targets?",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_BYTES,
                data=(
                    b"product,q3_revenue,q4_revenue,growth\n"
                    b"Widget A,150000,195000,30%\n"
                    b"Widget B,200000,220000,10%\n"
                ),
                name="revenue.csv",
            ),
            Attachment(
                type=AttachmentType.TEXT,
                data='{"targets": {"Widget A": 200000, "Widget B": 250000}}',
                name="targets.json",
            ),
        ],
    )

    print("  Attachments:")
    for att in task.attachments:
        ext = att.name.rsplit(".", 1)[-1] if att.name else "?"
        native = f".{ext}" in agent.llm.SUPPORTED_FILE_EXTENSIONS
        print(f"    - {att.type.value:15s}  {att.name:20s}  native={native}")

    result = await agent.execute(task)
    print(f"  Result:     {str(result)[:100]}...")


# ======================================================================
# 4. FILE_BASE64 - pre-encoded (no double-encoding)
# ======================================================================


async def example_file_base64():
    """
    FILE_BASE64: data is already base64-encoded.

    Common when receiving file data from APIs, databases, or webhooks.
    OpenAI provider passes the base64 directly -- no re-encoding.
    Framework-level decodes and extracts text.
    """
    print("\n" + "=" * 70)
    print("4. FILE_BASE64 (Pre-Encoded)")
    print("=" * 70)

    agent = Agent(
        name="APIBot",
        role="API processor",
        objective="Process API data",
        llm=get_llm(),
        prompt=ZeroShotPrompt().configure(
            system="You are a helpful assistant. Interpret file content supplied as base64 and answer the user's question.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    raw_content = b"id,name,department\n1,Alice,Engineering\n2,Bob,Marketing"
    b64_from_api = base64.b64encode(raw_content).decode()

    task = Task(
        id="b64-native",
        objective="List all employees by department.",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_BASE64,
                data=b64_from_api,
                name="employees.csv",
                mime_type="text/csv",
            ),
        ],
    )

    is_native = AttachmentType.FILE_BASE64 in agent.llm.NATIVE_ATTACHMENT_TYPES
    print(f"  Attachment: {task.attachments[0].name}")
    print(f"  Data:       (base64 string, {len(b64_from_api)} chars)")
    print(f"  Native:     {is_native}")

    result = await agent.execute(task)
    print(f"  Result:     {str(result)[:100]}...")


# ======================================================================
# 5. FILE_URL - remote file reference
# ======================================================================


async def example_file_url():
    """
    FILE_URL: model fetches the file server-side (Responses API).

    The file is NOT downloaded by the framework. The model
    accesses it directly via the URL.
    """
    print("\n" + "=" * 70)
    print("5. FILE_URL (Remote File Reference)")
    print("=" * 70)

    agent = Agent(
        name="RemoteBot",
        role="Document processor",
        objective="Process remote files",
        llm=get_llm(),
        prompt=ZeroShotPrompt().configure(
            system="You are a document assistant. Answer based on remotely referenced files when the model can access them.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(
        id="url-native",
        objective="Extract tables from this annual report.",
        attachments=[
            Attachment(
                type=AttachmentType.FILE_URL,
                data="https://storage.example.com/reports/annual_2025.pdf",
                name="annual_2025.pdf",
            ),
        ],
    )

    is_native = AttachmentType.FILE_URL in agent.llm.NATIVE_ATTACHMENT_TYPES
    print(f"  URL:        {task.attachments[0].data}")
    print(f"  Native:     {is_native}")
    if is_native:
        print("  Behaviour:  OpenAI Responses API fetches and processes server-side")
    else:
        print("  Behaviour:  Text fallback (file not downloaded by framework)")

    result = await agent.execute(task)
    print(f"  Result:     {str(result)[:100]}...")


# ======================================================================
# 6. Mixed files + images
# ======================================================================


async def example_mixed():
    """Mix any combination of attachment types in one task."""
    print("\n" + "=" * 70)
    print("6. Mixed Files + Images")
    print("=" * 70)

    agent = Agent(
        name="MultiBot",
        role="Research analyst",
        objective="Cross-reference sources",
        llm=get_llm(),
        prompt=ZeroShotPrompt().configure(
            system="You are a research analyst. Cross-check PDFs, images, spreadsheets, and URLs the user provides.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(
        id="mixed-native",
        objective="Compare the report with the chart and data. Find discrepancies.",
        attachments=[
            Attachment(
                type=AttachmentType.PDF,
                data=b"%PDF-1.4 report content",
                name="report.pdf",
            ),
            Attachment(
                type=AttachmentType.IMAGE_URL,
                data="https://example.com/charts/revenue.png",
                detail="high",
            ),
            Attachment(
                type=AttachmentType.FILE_BYTES,
                data=b"month,actual\nJan,100\nFeb,130\nMar,115",
                name="actuals.csv",
            ),
            Attachment(
                type=AttachmentType.FILE_URL,
                data="https://cdn.example.com/extra.xlsx",
                name="extra.xlsx",
            ),
        ],
    )

    print("  Attachments:")
    native_types = agent.llm.NATIVE_ATTACHMENT_TYPES
    for att in task.attachments:
        label = att.name or str(att.data)[:40]
        native = att.type in native_types
        print(f"    - {att.type.value:15s}  {label:25s}  native={native}")

    result = await agent.execute(task)
    print(f"  Result:     {str(result)[:100]}...")


# ======================================================================
# 7. Streaming with file attachments
# ======================================================================


async def example_streaming():
    """Streaming works identically with or without file attachments."""
    print("\n" + "=" * 70)
    print("7. Streaming with File Attachments")
    print("=" * 70)

    llm = get_llm()
    if isinstance(llm, MockLLM):
        llm = MockLLM(stream_chunk_size=8)

    agent = Agent(
        name="StreamBot",
        role="Writer",
        objective="Write summaries",
        llm=llm,
        prompt=ZeroShotPrompt().configure(
            system="You are a concise writer. Summarize attached notes clearly.",
        ),
        config=AgentConfig(execution_mode=ExecutionMode.DIRECT),
    )

    task = Task(
        id="stream-native",
        objective="Write a brief summary of these meeting notes.",
        attachments=[
            Attachment(
                type=AttachmentType.TEXT,
                data="Launch: Jan 15. Design: approved. Eng: 3 sprints left.",
                name="meeting_notes.txt",
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
# Main
# ======================================================================


async def main():
    print("NucleusIQ - OpenAI Native File Input")
    print("=" * 70)
    print()
    if USE_REAL_LLM:
        print("Running with REAL OpenAI API (OPENAI_API_KEY detected)")
    else:
        print("Running with MockLLM (set OPENAI_API_KEY for real responses)")
    print()
    print("Same Attachment API, smarter wire format with OpenAI.")
    print("Provider capabilities (mirrors NATIVE_TOOL_TYPES pattern):")
    print()
    print("  BaseOpenAI.NATIVE_ATTACHMENT_TYPES:")
    for t in sorted(BaseOpenAI.NATIVE_ATTACHMENT_TYPES, key=lambda x: x.value):
        print(f"    - {t.value}")
    print()
    print(
        f"  BaseOpenAI.SUPPORTED_FILE_EXTENSIONS: {len(BaseOpenAI.SUPPORTED_FILE_EXTENSIONS)} types"
    )
    print()

    await example_capabilities()
    await example_pdf_native()
    await example_structured_data()
    await example_file_base64()
    await example_file_url()
    await example_mixed()
    await example_streaming()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print()
    print("Key takeaway: You write Attachment code once.")
    print("The provider handles the optimization transparently.")
    print("Use llm.describe_attachment_support() to see what happens.")


if __name__ == "__main__":
    asyncio.run(main())
