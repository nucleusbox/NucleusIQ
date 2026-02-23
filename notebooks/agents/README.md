# NucleusIQ Agent Examples

End-to-end notebooks demonstrating how to build autonomous agents with NucleusIQ.

## Examples

| Notebook | Domain | What it shows |
|----------|--------|--------------|
| [PE Due Diligence](pe_due_diligence.ipynb) | Private Equity | 8 multi-step financial analyses — WACC, DCF, LBO IRR, merger math. Compares Standard vs Autonomous modes with external validation. |

## How to Run

1. **Install dependencies**:
   ```bash
   pip install nucleusiq nucleusiq-openai
   ```

2. **Set your API key**:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

3. **Open the notebook**:
   ```bash
   jupyter notebook notebooks/agents/pe_due_diligence.ipynb
   ```

## Adding New Examples

Each example notebook should follow this structure:

1. **Introduction** — What problem are we solving?
2. **Setup** — Imports and environment
3. **Tools** — Define domain-specific tools
4. **Tasks** — Define the scenarios with ground truth
5. **Standard Mode** — Run as baseline
6. **Autonomous Mode** — Run with validation plugins
7. **Results** — Compare and analyze
8. **Key Takeaways** — What we learned

Place new notebooks directly in this folder:

```
notebooks/
  agents/
    pe_due_diligence.ipynb
    research_analyst.ipynb      # future
    customer_support.ipynb      # future
    README.md
```
