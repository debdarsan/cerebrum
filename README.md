# Cerebrum - Multi-Agent Financial Intelligence System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Cerebrum is a sophisticated multi-agent financial intelligence system that combines specialized AI agents to provide comprehensive financial analysis. Unlike single-model approaches, Cerebrum uses a team of specialist agents that collaborate like human analysts to deliver deeper insights.

## 🧠 What is Cerebrum?

Imagine walking into a top-tier investment firm. You don't see one person doing everything. Instead, you find specialists: the quantitative analyst crunching numbers, the researcher diving into SEC filings, the risk manager scanning for threats, and the portfolio manager synthesizing insights.

Cerebrum recreates this specialist team structure in AI, where each agent brings unique expertise to solve complex financial puzzles.

## 🎯 Key Features

- **7 Specialized Agents**: Each agent masters a specific domain
- **Real Data Processing**: Works with actual SEC filings and financial databases
- **Multi-Tier Memory**: Episodic, semantic, and procedural memory systems
- **Dynamic Orchestration**: Intelligent agent selection based on query complexity
- **Framework Agnostic**: No dependency on LangChain or similar frameworks

## 🤖 The Agent Team

- **DataMiner**: Document analysis and information extraction
- **Quant**: Numerical analysis and database operations  
- **TrendScout**: Temporal analysis and pattern recognition
- **RiskAssessor**: Risk identification and assessment
- **ComplianceWatcher**: Regulatory compliance analysis
- **MarketPulse**: Market sentiment and real-time intelligence
- **VisionAnalyst**: Visual data interpretation and chart analysis

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/cerebrum.git
cd cerebrum
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from cerebrum import CerebrumOrchestrator
from cerebrum.core.agents import *
from cerebrum.core.memory import MemoryKeeper

# Initialize the system
memory = MemoryKeeper()
agents = {
    'DataMiner': DataMinerAgent(llm_client, vector_store, embedding_model),
    'Quant': QuantAgent(llm_client, database_path),
    'TrendScout': TrendScoutAgent(llm_client, database_path),
    'RiskAssessor': RiskAssessorAgent(llm_client, vector_store),
    'ComplianceWatcher': ComplianceWatcherAgent(llm_client),
    'MarketPulse': MarketPulseAgent(llm_client, web_search_api_key),
    'VisionAnalyst': VisionAnalystAgent(llm_client)
}

cerebrum = CerebrumOrchestrator(agents, memory, llm_client)

# Analyze a complex financial question
result = cerebrum.process_request(
    "Analyze Microsoft's revenue growth and assess related competitive risks"
)

print(result['final_analysis'])
```


## 🔧 Configuration

Set up your environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional: for web search
```

Or create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
DATABASE_PATH=financials.db
```

## 🔬 Examples

### Financial Analysis
```python
# Comprehensive analysis
result = cerebrum.process_request(
    "Analyze Q3 earnings performance and competitive positioning"
)
```

### Risk Assessment
```python
# Risk-focused analysis
result = cerebrum.process_request(
    "Identify and assess top 5 financial risks from latest 10-K filing"
)
```

### Trend Analysis
```python
# Time-series analysis
result = cerebrum.process_request(
    "Compare revenue growth trends across business segments over last 2 years"
)
```


## 🔒 Security & Privacy

- No data persistence by default (in-memory processing)
- API keys managed through environment variables
- Optional local-only mode (no external API calls)
- Audit trail for all agent decisions

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Microsoft Corporation for publicly available SEC filing data
- OpenAI for GPT-4 API
- The open-source community for foundational libraries

## 🗺️ Roadmap

- [ ] Web UI dashboard
- [ ] Additional data source connectors
- [ ] Real-time streaming analysis
- [ ] Custom agent marketplace
- [ ] Enterprise deployment tools

---

**Built with ❤️ for the financial analysis community**
