"""
Cerebrum: A Multi-Agent Financial Intelligence System
======================================================
A sophisticated AI system that combines specialized agents for comprehensive financial analysis.

Author: [Your Name]
License: MIT
Python Version: 3.8+
"""

import os
import json
import sqlite3
import hashlib
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import yfinance as yf
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Cerebrum')


# ============================================================================
# Configuration and Constants
# ============================================================================

class AgentType(Enum):
    """Types of specialist agents in the system"""
    DATA_MINER = "DataMiner"
    QUANT = "Quant"
    TREND_SCOUT = "TrendScout"
    RISK_ASSESSOR = "RiskAssessor"
    COMPLIANCE_WATCHER = "ComplianceWatcher"
    MARKET_PULSE = "MarketPulse"
    VISION_ANALYST = "VisionAnalyst"


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CerebrumConfig:
    """Central configuration for the Cerebrum system"""
    openai_api_key: str
    database_path: str = "financials.db"
    vector_db_path: str = "./cerebrum_vectors"
    memory_db_path: str = "./cerebrum_memory.json"
    max_agents_per_request: int = 5
    confidence_threshold: float = 0.7
    enable_proactive_monitoring: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Create necessary directories
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.memory_db_path)).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Core State Management
# ============================================================================

@dataclass
class CerebrumState:
    """Maintains the shared state across all agents during analysis"""
    request: str = ""
    plan: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    risks_identified: List[Dict[str, Any]] = field(default_factory=list)
    compliance_issues: List[Dict[str, Any]] = field(default_factory=list)
    final_report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_finding(self, agent: str, finding: Dict[str, Any]):
        """Add a finding from an agent"""
        finding['timestamp'] = datetime.now().isoformat()
        finding['agent'] = agent
        self.findings.append(finding)
        
        if 'confidence' in finding:
            self.confidence_scores.append(finding['confidence'])
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all findings"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


# ============================================================================
# Base Agent Architecture
# ============================================================================

class BaseAgent:
    """Base class for all specialist agents"""
    
    def __init__(self, name: str, llm_client: OpenAI, config: CerebrumConfig):
        self.name = name
        self.llm_client = llm_client
        self.config = config
        self.specialty = "General Analysis"
        self.logger = logging.getLogger(f'Cerebrum.{name}')
    
    def invoke_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Safely invoke the LLM with error handling"""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {str(e)}")
            return ""
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process a task and return findings. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate agent output meets minimum requirements"""
        required_fields = ['agent', 'task', 'method', 'confidence']
        return all(field in output for field in required_fields)


# ============================================================================
# DataMiner Agent - Document Intelligence
# ============================================================================

class DataMinerAgent(BaseAgent):
    """Specialist in document analysis and information extraction"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("DataMiner", llm_client, config)
        self.specialty = "Document Analysis & Information Retrieval"
        self.init_vector_store()
    
    def init_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.config.vector_db_path)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.openai_api_key,
                model_name="text-embedding-ada-002"
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name="financial_documents",
                    embedding_function=self.embedding_function
                )
            except:
                self.collection = self.chroma_client.create_collection(
                    name="financial_documents",
                    embedding_function=self.embedding_function
                )
        except Exception as e:
            self.logger.warning(f"Vector store initialization failed: {e}")
            self.collection = None
    
    def enhance_query(self, query: str) -> str:
        """Transform vague queries into precise search terms"""
        prompt = f"""
        You're a financial research specialist. Transform this query into specific search terms.
        Focus on financial terminology, metrics, and concepts.
        
        Original query: {query}
        
        Enhanced search terms (comma-separated):
        """
        return self.invoke_llm(prompt, temperature=0.3)
    
    def extract_key_insights(self, documents: List[str], query: str) -> List[str]:
        """Extract the most relevant insights from documents"""
        if not documents:
            return []
        
        doc_context = "\n\n".join(documents[:5])  # Limit to top 5 documents
        
        prompt = f"""
        Extract key insights relevant to this query from the following documents:
        
        Query: {query}
        
        Documents:
        {doc_context[:3000]}  # Limit context size
        
        Provide 3-5 bullet points with specific, actionable insights:
        """
        
        insights_text = self.invoke_llm(prompt, temperature=0.5)
        return [insight.strip() for insight in insights_text.split('\n') if insight.strip()]
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process document analysis task"""
        self.logger.info(f"🔍 Analyzing documents for: {task}")
        
        try:
            # Enhance the search query
            enhanced_query = self.enhance_query(task)
            
            results = []
            insights = []
            
            if self.collection:
                # Perform semantic search
                search_results = self.collection.query(
                    query_texts=[enhanced_query],
                    n_results=10
                )
                
                if search_results['documents'][0]:
                    # Extract insights from retrieved documents
                    insights = self.extract_key_insights(
                        search_results['documents'][0],
                        task
                    )
                    
                    # Format results
                    for i, doc in enumerate(search_results['documents'][0][:5]):
                        results.append({
                            'content': doc[:500],
                            'relevance_score': 1.0 - (search_results['distances'][0][i] if search_results['distances'] else 0.5),
                            'metadata': search_results['metadatas'][0][i] if i < len(search_results['metadatas'][0]) else {}
                        })
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'semantic_document_search',
                'documents_analyzed': len(results),
                'search_terms': enhanced_query,
                'results': results,
                'insights': insights,
                'confidence': 0.85 if results else 0.3
            }
            
            state.add_finding(self.name, output)
            return output
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'semantic_document_search',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# Quant Agent - Numerical Analysis
# ============================================================================

class QuantAgent(BaseAgent):
    """Specialist in quantitative analysis and database operations"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("Quant", llm_client, config)
        self.specialty = "Quantitative Analysis & Database Operations"
        self.db_path = config.database_path
    
    def get_database_schema(self) -> str:
        """Retrieve database schema information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema_info.append(f"Table: {table_name}")
                for col in columns:
                    schema_info.append(f"  - {col[1]} ({col[2]})")
            
            conn.close()
            return "\n".join(schema_info)
        except Exception as e:
            self.logger.error(f"Failed to get schema: {e}")
            return "Schema unavailable"
    
    def generate_sql(self, question: str, schema: str) -> str:
        """Convert natural language to SQL query"""
        prompt = f"""
        You're a SQL expert. Generate a precise SQL query for this question.
        Return ONLY the SQL query, no explanations.
        
        Database Schema:
        {schema}
        
        Question: {question}
        
        SQL Query:
        """
        return self.invoke_llm(prompt, temperature=0.2).strip()
    
    def analyze_results(self, query: str, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on query results"""
        analysis = {
            'row_count': len(results_df),
            'columns': list(results_df.columns),
        }
        
        # Analyze numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            analysis[f'{col}_stats'] = {
                'mean': float(results_df[col].mean()),
                'median': float(results_df[col].median()),
                'std': float(results_df[col].std()),
                'min': float(results_df[col].min()),
                'max': float(results_df[col].max())
            }
        
        return analysis
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process quantitative analysis task"""
        self.logger.info(f"📊 Running quantitative analysis for: {task}")
        
        try:
            # Get database schema
            schema = self.get_database_schema()
            
            # Generate SQL query
            sql_query = self.generate_sql(task, schema)
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            results_df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            # Analyze results
            statistical_analysis = self.analyze_results(sql_query, results_df)
            
            # Generate insights
            insights_prompt = f"""
            Analyze these query results and provide key financial insights:
            
            Query: {sql_query}
            Results Summary: {statistical_analysis}
            Sample Data: {results_df.head().to_string() if len(results_df) > 0 else "No data"}
            
            Provide 3-5 specific, quantitative insights:
            """
            
            insights = self.invoke_llm(insights_prompt, temperature=0.5)
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'sql_quantitative_analysis',
                'sql_query': sql_query,
                'row_count': len(results_df),
                'statistical_analysis': statistical_analysis,
                'sample_data': results_df.head().to_dict() if len(results_df) > 0 else {},
                'insights': insights,
                'confidence': 0.9 if len(results_df) > 0 else 0.3
            }
            
            state.add_finding(self.name, output)
            return output
            
        except Exception as e:
            self.logger.error(f"Quantitative analysis failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'sql_quantitative_analysis',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# TrendScout Agent - Temporal Pattern Analysis
# ============================================================================

class TrendScoutAgent(BaseAgent):
    """Specialist in temporal analysis and trend identification"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("TrendScout", llm_client, config)
        self.specialty = "Trend Analysis & Temporal Patterns"
        self.db_path = config.database_path
    
    def calculate_growth_metrics(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Calculate comprehensive growth metrics"""
        df = df.copy()
        
        # Period-over-period changes
        df[f'{value_col}_change'] = df[value_col].diff()
        df[f'{value_col}_pct_change'] = df[value_col].pct_change() * 100
        
        # Moving averages
        df[f'{value_col}_ma3'] = df[value_col].rolling(3, min_periods=1).mean()
        df[f'{value_col}_ma6'] = df[value_col].rolling(6, min_periods=1).mean()
        
        # Volatility
        df[f'{value_col}_volatility'] = df[value_col].rolling(3, min_periods=1).std()
        
        # Trend direction
        df['trend_direction'] = np.where(
            df[f'{value_col}_change'] > 0, 'up',
            np.where(df[f'{value_col}_change'] < 0, 'down', 'flat')
        )
        
        return df
    
    def identify_patterns(self, df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
        """Identify specific patterns in time series data"""
        patterns = {
            'trend_type': 'unknown',
            'seasonality_detected': False,
            'anomalies': [],
            'turning_points': []
        }
        
        if len(df) < 3:
            return patterns
        
        # Overall trend
        correlation = df.index.to_series().corr(df[value_col])
        if correlation > 0.7:
            patterns['trend_type'] = 'strong_upward'
        elif correlation > 0.3:
            patterns['trend_type'] = 'moderate_upward'
        elif correlation < -0.7:
            patterns['trend_type'] = 'strong_downward'
        elif correlation < -0.3:
            patterns['trend_type'] = 'moderate_downward'
        else:
            patterns['trend_type'] = 'sideways'
        
        # Detect anomalies (values beyond 2 standard deviations)
        mean_val = df[value_col].mean()
        std_val = df[value_col].std()
        
        anomalies = df[
            (df[value_col] > mean_val + 2 * std_val) |
            (df[value_col] < mean_val - 2 * std_val)
        ]
        
        if not anomalies.empty:
            patterns['anomalies'] = anomalies.index.tolist()
        
        # Detect turning points
        if f'{value_col}_change' in df.columns:
            sign_changes = df[f'{value_col}_change'].diff().fillna(0)
            turning_points = df[sign_changes != 0]
            if not turning_points.empty:
                patterns['turning_points'] = turning_points.index.tolist()[:5]  # Top 5
        
        return patterns
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process trend analysis task"""
        self.logger.info(f"📈 Analyzing trends for: {task}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Try to load time-series data
            # First, check what tables exist
            tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables_df = pd.read_sql_query(tables_query, conn)
            
            # Look for tables with temporal data
            time_series_data = None
            table_used = None
            
            for table_name in tables_df['name']:
                # Check if table has date/time columns
                schema_query = f"PRAGMA table_info({table_name});"
                schema_df = pd.read_sql_query(schema_query, conn)
                
                time_columns = schema_df[
                    schema_df['name'].str.contains('date|time|year|quarter|month', case=False)
                ]['name'].tolist()
                
                if time_columns:
                    # Found a table with temporal data
                    query = f"SELECT * FROM {table_name} LIMIT 1000"
                    time_series_data = pd.read_sql_query(query, conn)
                    table_used = table_name
                    break
            
            conn.close()
            
            if time_series_data is None or time_series_data.empty:
                raise ValueError("No temporal data found in database")
            
            # Identify numeric columns for analysis
            numeric_cols = time_series_data.select_dtypes(include=[np.number]).columns
            
            trend_analyses = {}
            patterns_identified = {}
            
            for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
                # Calculate growth metrics
                analyzed_df = self.calculate_growth_metrics(time_series_data, col)
                
                # Identify patterns
                patterns = self.identify_patterns(analyzed_df, col)
                
                trend_analyses[col] = {
                    'latest_value': float(analyzed_df[col].iloc[-1]) if len(analyzed_df) > 0 else 0,
                    'average_growth': float(analyzed_df[f'{col}_pct_change'].mean()) if f'{col}_pct_change' in analyzed_df else 0,
                    'volatility': float(analyzed_df[f'{col}_volatility'].mean()) if f'{col}_volatility' in analyzed_df else 0,
                    'trend_direction': patterns['trend_type']
                }
                
                patterns_identified[col] = patterns
            
            # Generate insights
            insights_prompt = f"""
            Analyze these trend patterns and provide insights:
            
            Trend Analysis: {json.dumps(trend_analyses, indent=2)}
            Patterns: {json.dumps(patterns_identified, indent=2)}
            
            Provide 3-5 specific insights about trends, seasonality, and future projections:
            """
            
            insights = self.invoke_llm(insights_prompt, temperature=0.5)
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'temporal_pattern_analysis',
                'table_analyzed': table_used,
                'time_periods_analyzed': len(time_series_data),
                'metrics_analyzed': list(trend_analyses.keys()),
                'trend_analyses': trend_analyses,
                'patterns_identified': patterns_identified,
                'insights': insights,
                'confidence': 0.88
            }
            
            state.add_finding(self.name, output)
            return output
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'temporal_pattern_analysis',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# RiskAssessor Agent - Risk Identification and Assessment
# ============================================================================

class RiskAssessorAgent(BaseAgent):
    """Specialist in risk identification and assessment"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("RiskAssessor", llm_client, config)
        self.specialty = "Risk Analysis & Threat Assessment"
        
        self.risk_categories = [
            "market_risk", "credit_risk", "operational_risk",
            "regulatory_risk", "liquidity_risk", "reputation_risk",
            "strategic_risk", "technology_risk", "geopolitical_risk"
        ]
    
    def assess_risk_level(self, risk_description: str) -> Dict[str, Any]:
        """Quantify risk level from description"""
        prompt = f"""
        Analyze this risk and provide a structured assessment:
        
        Risk: {risk_description}
        
        Return a JSON response with these exact fields:
        {{
            "risk_level": "low|medium|high|critical",
            "probability": 0.0-1.0,
            "impact": "minimal|moderate|significant|severe",
            "timeframe": "immediate|short_term|medium_term|long_term",
            "category": "market|credit|operational|regulatory|liquidity|reputation|strategic|technology|geopolitical",
            "mitigation_difficulty": "easy|moderate|difficult|very_difficult"
        }}
        
        JSON:
        """
        
        try:
            response = self.invoke_llm(prompt, temperature=0.3)
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "risk_level": "medium",
                    "probability": 0.5,
                    "impact": "moderate",
                    "timeframe": "medium_term",
                    "category": "operational"
                }
        except Exception as e:
            self.logger.warning(f"Risk assessment parsing failed: {e}")
            return {
                "risk_level": "medium",
                "probability": 0.5,
                "impact": "moderate",
                "timeframe": "medium_term",
                "category": "operational"
            }
    
    def generate_risk_scenarios(self, context: str) -> List[Dict[str, Any]]:
        """Generate potential risk scenarios based on context"""
        prompt = f"""
        Based on this context, identify the top 5 potential risks:
        
        Context: {context}
        
        For each risk, provide:
        1. Risk description (one sentence)
        2. Potential trigger events
        3. Impact on business
        4. Early warning indicators
        
        Format as numbered list:
        """
        
        response = self.invoke_llm(prompt, temperature=0.6)
        
        # Parse response into structured risks
        risks = []
        risk_blocks = response.split('\n\n')
        
        for block in risk_blocks[:5]:
            if block.strip():
                risks.append({
                    'description': block.split('\n')[0] if block else "Unknown risk",
                    'full_analysis': block
                })
        
        return risks
    
    def calculate_risk_score(self, risk_assessment: Dict[str, Any]) -> float:
        """Calculate composite risk score"""
        # Probability weight
        probability = risk_assessment.get('probability', 0.5)
        
        # Impact scoring
        impact_scores = {
            'minimal': 0.25,
            'moderate': 0.5,
            'significant': 0.75,
            'severe': 1.0
        }
        impact_score = impact_scores.get(risk_assessment.get('impact', 'moderate'), 0.5)
        
        # Timeframe urgency
        timeframe_scores = {
            'immediate': 1.0,
            'short_term': 0.75,
            'medium_term': 0.5,
            'long_term': 0.25
        }
        timeframe_score = timeframe_scores.get(risk_assessment.get('timeframe', 'medium_term'), 0.5)
        
        # Composite score
        risk_score = (probability * 0.4 + impact_score * 0.4 + timeframe_score * 0.2)
        
        return round(risk_score, 2)
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process risk assessment task"""
        self.logger.info(f"⚠️ Assessing risks for: {task}")
        
        try:
            # Generate risk scenarios
            risk_scenarios = self.generate_risk_scenarios(task)
            
            # Assess each risk
            assessed_risks = []
            for scenario in risk_scenarios:
                assessment = self.assess_risk_level(scenario['description'])
                assessment['description'] = scenario['description']
                assessment['risk_score'] = self.calculate_risk_score(assessment)
                assessed_risks.append(assessment)
            
            # Sort by risk score
            assessed_risks.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # Categorize risks
            high_priority = [r for r in assessed_risks if r['risk_level'] in ['high', 'critical']]
            medium_priority = [r for r in assessed_risks if r['risk_level'] == 'medium']
            low_priority = [r for r in assessed_risks if r['risk_level'] == 'low']
            
            # Generate mitigation recommendations
            if high_priority:
                mitigation_prompt = f"""
                For these high-priority risks, suggest specific mitigation strategies:
                
                Risks:
                {json.dumps([r['description'] for r in high_priority[:3]], indent=2)}
                
                Provide actionable mitigation strategies:
                """
                
                mitigation_strategies = self.invoke_llm(mitigation_prompt, temperature=0.5)
            else:
                mitigation_strategies = "No high-priority risks identified requiring immediate mitigation."
            
            # Calculate overall risk profile
            avg_risk_score = sum(r['risk_score'] for r in assessed_risks) / len(assessed_risks) if assessed_risks else 0
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'comprehensive_risk_assessment',
                'total_risks_identified': len(assessed_risks),
                'high_priority_count': len(high_priority),
                'medium_priority_count': len(medium_priority),
                'low_priority_count': len(low_priority),
                'average_risk_score': round(avg_risk_score, 2),
                'risk_breakdown': assessed_risks,
                'mitigation_strategies': mitigation_strategies,
                'risk_categories_covered': list(set(r['category'] for r in assessed_risks)),
                'insights': f"Identified {len(assessed_risks)} risks with average score {avg_risk_score:.2f}. {len(high_priority)} require immediate attention.",
                'confidence': 0.82
            }
            
            # Update state with risks
            state.risks_identified.extend(high_priority)
            state.add_finding(self.name, output)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'comprehensive_risk_assessment',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# ComplianceWatcher Agent - Regulatory Monitoring
# ============================================================================

class ComplianceWatcherAgent(BaseAgent):
    """Specialist in regulatory compliance and legal requirements"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("ComplianceWatcher", llm_client, config)
        self.specialty = "Regulatory Compliance & Legal Analysis"
        
        self.compliance_frameworks = {
            'SEC': ['10-K', '10-Q', '8-K', 'proxy statements', 'insider trading'],
            'SOX': ['internal controls', 'financial reporting', 'audit requirements'],
            'GDPR': ['data privacy', 'consent', 'right to deletion', 'data portability'],
            'FCPA': ['anti-bribery', 'books and records', 'internal controls'],
            'Basel_III': ['capital requirements', 'leverage ratio', 'liquidity coverage'],
            'MiFID_II': ['best execution', 'transparency', 'investor protection']
        }
    
    def check_compliance_requirements(self, context: str) -> List[Dict[str, Any]]:
        """Identify applicable compliance requirements"""
        prompt = f"""
        Review this business context and identify all applicable compliance requirements:
        
        Context: {context}
        
        For each requirement, specify:
        1. Regulation/Framework (e.g., SEC, SOX, GDPR)
        2. Specific requirement
        3. Current status (if determinable)
        4. Potential penalties for non-compliance
        5. Recommended actions
        
        Format as a numbered list:
        """
        
        response = self.invoke_llm(prompt, temperature=0.3)
        
        # Parse into structured requirements
        requirements = []
        for line in response.split('\n'):
            if line.strip() and any(framework in line for framework in self.compliance_frameworks.keys()):
                requirements.append({
                    'requirement': line.strip(),
                    'framework': next((f for f in self.compliance_frameworks.keys() if f in line), 'General')
                })
        
        return requirements
    
    def assess_compliance_gaps(self, requirements: List[Dict[str, Any]], context: str) -> Dict[str, Any]:
        """Assess gaps in compliance"""
        if not requirements:
            return {'gaps': [], 'compliance_score': 100}
        
        prompt = f"""
        Assess potential compliance gaps based on these requirements and context:
        
        Requirements:
        {json.dumps(requirements, indent=2)}
        
        Context: {context}
        
        Identify any gaps, rate their severity (low/medium/high), and suggest remediation.
        
        Format as JSON:
        {{
            "gaps": [
                {{
                    "area": "specific compliance area",
                    "severity": "low|medium|high",
                    "description": "gap description",
                    "remediation": "suggested action"
                }}
            ],
            "overall_compliance_health": "good|fair|poor"
        }}
        
        JSON:
        """
        
        try:
            response = self.invoke_llm(prompt, temperature=0.3)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Calculate compliance score
                gap_count = len(result.get('gaps', []))
                high_severity_gaps = sum(1 for g in result.get('gaps', []) if g.get('severity') == 'high')
                
                compliance_score = max(0, 100 - (gap_count * 10) - (high_severity_gaps * 15))
                result['compliance_score'] = compliance_score
                
                return result
            else:
                return {'gaps': [], 'compliance_score': 75}
        except Exception as e:
            self.logger.warning(f"Gap assessment parsing failed: {e}")
            return {'gaps': [], 'compliance_score': 75}
    
    def generate_compliance_calendar(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate compliance calendar with key dates"""
        calendar_items = []
        
        # Common compliance deadlines (examples)
        standard_deadlines = {
            '10-K': 'Annual - 60-90 days after fiscal year end',
            '10-Q': 'Quarterly - 40-45 days after quarter end',
            '8-K': 'Within 4 business days of triggering event',
            'proxy statements': '120 days before annual meeting',
            'SOX certification': 'With each 10-K and 10-Q filing',
            'GDPR review': 'Annual privacy impact assessment'
        }
        
        for req in requirements:
            for deadline_key, deadline_desc in standard_deadlines.items():
                if deadline_key.lower() in req.get('requirement', '').lower():
                    calendar_items.append({
                        'item': deadline_key,
                        'frequency': deadline_desc,
                        'framework': req.get('framework'),
                        'importance': 'high'
                    })
        
        return calendar_items
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process compliance monitoring task"""
        self.logger.info(f"⚖️ Monitoring compliance for: {task}")
        
        try:
            # Identify applicable compliance requirements
            requirements = self.check_compliance_requirements(task)
            
            # Assess compliance gaps
            gap_assessment = self.assess_compliance_gaps(requirements, task)
            
            # Generate compliance calendar
            compliance_calendar = self.generate_compliance_calendar(requirements)
            
            # Identify frameworks involved
            frameworks_involved = list(set(r.get('framework') for r in requirements))
            
            # Generate compliance recommendations
            if gap_assessment.get('gaps'):
                recommendations_prompt = f"""
                Based on these compliance gaps, provide prioritized recommendations:
                
                Gaps: {json.dumps(gap_assessment.get('gaps', []), indent=2)}
                
                Provide 3-5 specific, actionable recommendations:
                """
                recommendations = self.invoke_llm(recommendations_prompt, temperature=0.4)
            else:
                recommendations = "No significant compliance gaps identified. Maintain current compliance practices."
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'compliance_review',
                'frameworks_analyzed': frameworks_involved,
                'requirements_identified': len(requirements),
                'compliance_score': gap_assessment.get('compliance_score', 75),
                'gaps_found': len(gap_assessment.get('gaps', [])),
                'gap_details': gap_assessment.get('gaps', []),
                'compliance_calendar': compliance_calendar[:5],  # Top 5 calendar items
                'recommendations': recommendations,
                'overall_health': gap_assessment.get('overall_compliance_health', 'fair'),
                'insights': f"Compliance score: {gap_assessment.get('compliance_score', 75)}/100. {len(gap_assessment.get('gaps', []))} gaps identified across {len(frameworks_involved)} regulatory frameworks.",
                'confidence': 0.85
            }
            
            # Update state with compliance issues
            state.compliance_issues.extend(gap_assessment.get('gaps', []))
            state.add_finding(self.name, output)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Compliance monitoring failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'compliance_review',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# MarketPulse Agent - Real-time Market Sentiment
# ============================================================================

class MarketPulseAgent(BaseAgent):
    """Specialist in real-time market sentiment and external factors"""
    
    def __init__(self, llm_client: OpenAI, config: CerebrumConfig):
        super().__init__("MarketPulse", llm_client, config)
        self.specialty = "Market Sentiment & Real-time Analysis"
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent price data
            hist = ticker.history(period="1mo")
            
            market_data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'volume': info.get('volume', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'month_change': ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100) if not hist.empty else 0,
                'volatility': hist['Close'].std() if not hist.empty else 0
            }
            
            return market_data
        except Exception as e:
            self.logger.warning(f"Failed to fetch market data for {symbol}: {e}")
            return {}
    
    def analyze_sentiment(self, context: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment based on various factors"""
        prompt = f"""
        Analyze market sentiment based on this information:
        
        Context: {context}
        Market Data: {json.dumps(market_data, indent=2)}
        
        Assess:
        1. Overall sentiment (bullish/neutral/bearish)
        2. Sentiment strength (1-10)
        3. Key drivers
        4. Market risks
        5. Opportunities
        
        Format as JSON:
        {{
            "sentiment": "bullish|neutral|bearish",
            "strength": 1-10,
            "drivers": ["driver1", "driver2"],
            "risks": ["risk1", "risk2"],
            "opportunities": ["opportunity1", "opportunity2"]
        }}
        
        JSON:
        """
        
        try:
            response = self.invoke_llm(prompt, temperature=0.4)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "sentiment": "neutral",
                    "strength": 5,
                    "drivers": [],
                    "risks": [],
                    "opportunities": []
                }
        except Exception as e:
            self.logger.warning(f"Sentiment analysis parsing failed: {e}")
            return {
                "sentiment": "neutral",
                "strength": 5,
                "drivers": [],
                "risks": [],
                "opportunities": []
            }
    
    def identify_market_catalysts(self, context: str) -> List[Dict[str, Any]]:
        """Identify potential market-moving catalysts"""
        prompt = f"""
        Identify potential market catalysts based on this context:
        
        Context: {context}
        
        List potential catalysts that could impact the market/stock:
        - Earnings announcements
        - Product launches
        - Regulatory decisions
        - Economic data releases
        - Competitor actions
        - Macro events
        
        For each catalyst, specify:
        1. Event description
        2. Expected timing
        3. Potential impact (positive/negative/mixed)
        4. Probability (low/medium/high)
        
        Provide as numbered list:
        """
        
        response = self.invoke_llm(prompt, temperature=0.5)
        
        catalysts = []
        for line in response.split('\n'):
            if line.strip() and any(char.isdigit() for char in line[:3]):
                catalysts.append({
                    'description': line.strip(),
                    'impact': 'unknown'
                })
        
        return catalysts[:10]  # Top 10 catalysts
    
    def process(self, task: str, state: CerebrumState) -> Dict[str, Any]:
        """Process market sentiment analysis task"""
        self.logger.info(f"📡 Analyzing market pulse for: {task}")
        
        try:
            # Extract ticker symbols from task (if any)
            import re
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            potential_tickers = re.findall(ticker_pattern, task)
            
            # Default to major indices if no ticker found
            tickers = potential_tickers[:3] if potential_tickers else ['SPY', 'QQQ', 'DIA']
            
            # Fetch market data
            market_data_collection = {}
            for ticker in tickers:
                data = self.get_market_data(ticker)
                if data:
                    market_data_collection[ticker] = data
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_sentiment(task, market_data_collection)
            
            # Identify market catalysts
            catalysts = self.identify_market_catalysts(task)
            
            # Calculate market momentum
            if market_data_collection:
                avg_monthly_change = sum(
                    data.get('month_change', 0) for data in market_data_collection.values()
                ) / len(market_data_collection)
                
                momentum = "positive" if avg_monthly_change > 2 else "negative" if avg_monthly_change < -2 else "neutral"
            else:
                avg_monthly_change = 0
                momentum = "neutral"
            
            # Generate market insights
            insights_prompt = f"""
            Based on this market analysis, provide key insights:
            
            Sentiment: {sentiment_analysis}
            Market Data: {json.dumps(market_data_collection, indent=2)}
            Catalysts: {catalysts[:3]}
            
            Provide 3-5 actionable market insights:
            """
            
            insights = self.invoke_llm(insights_prompt, temperature=0.5)
            
            output = {
                'agent': self.name,
                'task': task,
                'method': 'market_sentiment_analysis',
                'tickers_analyzed': list(market_data_collection.keys()),
                'sentiment': sentiment_analysis.get('sentiment', 'neutral'),
                'sentiment_strength': sentiment_analysis.get('strength', 5),
                'market_momentum': momentum,
                'avg_monthly_change': round(avg_monthly_change, 2),
                'key_drivers': sentiment_analysis.get('drivers', []),
                'market_risks': sentiment_analysis.get('risks', []),
                'opportunities': sentiment_analysis.get('opportunities', []),
                'upcoming_catalysts': catalysts[:5],
                'market_data': market_data_collection,
                'insights': insights,
                'confidence': 0.78
            }
            
            state.add_finding(self.name, output)
            return output
            
        except Exception as e:
            self.logger.error(f"Market sentiment analysis failed: {e}")
            error_output = {
                'agent': self.name,
                'task': task,
                'method': 'market_sentiment_analysis',
                'error': str(e),
                'confidence': 0.1
            }
            state.add_finding(self.name, error_output)
            return error_output


# ============================================================================
# Memory System - Institutional Knowledge
# ============================================================================

class MemoryKeeper:
    """Persistent memory system for institutional knowledge"""
    
    def __init__(self, memory_path: str = "./cerebrum_memory.json"):
        self.memory_path = memory_path
        self.short_term = {}  # Current session memory
        self.long_term = {}   # Persistent memory
        self.insights_db = {}  # Key insights database
        self.access_patterns = {}  # Track usage patterns
        
        self.load_memory()
    
    def load_memory(self):
        """Load persistent memory from disk"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                    self.long_term = data.get('long_term', {})
                    self.insights_db = data.get('insights_db', {})
                    self.access_patterns = data.get('access_patterns', {})
                logger.info(f"Loaded {len(self.long_term)} memories from disk")
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
    
    def save_memory(self):
        """Save persistent memory to disk"""
        try:
            data = {
                'long_term': self.long_term,
                'insights_db': self.insights_db,
                'access_patterns': self.access_patterns
            }
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.long_term)} memories to disk")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def store_insight(self, key: str, insight: Any, category: str = "general", importance: float = 0.5):
        """Store a key insight for future reference"""
        memory_entry = {
            'insight': insight,
            'category': category,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0,
            'last_accessed': None
        }
        
        self.long_term[key] = memory_entry
        
        # Update category index
        if category not in self.insights_db:
            self.insights_db[category] = []
        if key not in self.insights_db[category]:
            self.insights_db[category].append(key)
        
        self.save_memory()
        logger.info(f"🧠 Stored insight: {key} (category: {category})")
    
    def recall_insights(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant stored insights"""
        relevant_insights = []
        query_words = set(query.lower().split())
        
        # Search through long-term memory
        for key, data in self.long_term.items():
            if category and data.get('category') != category:
                continue
            
            # Calculate relevance
            key_words = set(key.lower().replace('_', ' ').split())
            insight_words = set(str(data.get('insight', '')).lower().split()[:50])  # First 50 words
            
            # Relevance scoring
            key_relevance = len(query_words & key_words)
            content_relevance = len(query_words & insight_words)
            relevance = key_relevance * 2 + content_relevance  # Weight key matches higher
            
            if relevance > 0:
                # Update access metrics
                data['access_count'] = data.get('access_count', 0) + 1
                data['last_accessed'] = datetime.now().isoformat()
                
                relevant_insights.append({
                    'key': key,
                    'insight': data['insight'],
                    'category': data['category'],
                    'relevance': relevance,
                    'importance': data.get('importance', 0.5),
                    'timestamp': data['timestamp'],
                    'access_count': data['access_count']
                })
        
        # Sort by relevance and importance
        relevant_insights.sort(
            key=lambda x: (x['relevance'] * x['importance']), 
            reverse=True
        )
        
        self.save_memory()
        return relevant_insights[:limit]
    
    def forget_old_memories(self, days_threshold: int = 90, importance_threshold: float = 0.3):
        """Remove old, unimportant memories to prevent memory bloat"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        keys_to_remove = []
        
        for key, data in self.long_term.items():
            # Check if memory is old and unimportant
            timestamp = datetime.fromisoformat(data['timestamp'])
            importance = data.get('importance', 0.5)
            access_count = data.get('access_count', 0)
            
            if (timestamp < cutoff_date and 
                importance < importance_threshold and 
                access_count < 5):
                keys_to_remove.append(key)
        
        # Remove old memories
        for key in keys_to_remove:
            del self.long_term[key]
            # Remove from category index
            for category, keys in self.insights_db.items():
                if key in keys:
                    keys.remove(key)
        
        if keys_to_remove:
            logger.info(f"Forgot {len(keys_to_remove)} old memories")
            self.save_memory()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        total_memories = len(self.long_term)
        categories = list(self.insights_db.keys())
        
        # Calculate average access count
        total_accesses = sum(m.get('access_count', 0) for m in self.long_term.values())
        avg_access = total_accesses / total_memories if total_memories > 0 else 0
        
        # Find most accessed memories
        most_accessed = sorted(
            self.long_term.items(),
            key=lambda x: x[1].get('access_count', 0),
            reverse=True
        )[:5]
        
        return {
            'total_memories': total_memories,
            'categories': categories,
            'average_access_count': avg_access,
            'most_accessed': [{'key': k, 'count': v.get('access_count', 0)} for k, v in most_accessed],
            'memory_size_bytes': os.path.getsize(self.memory_path) if os.path.exists(self.memory_path) else 0
        }


# ============================================================================
# Orchestrator - Main Coordinator
# ============================================================================

class CerebrumOrchestrator:
    """Main coordinator that manages all specialist agents"""
    
    def __init__(self, config: CerebrumConfig):
        self.config = config
        self.llm_client = OpenAI(api_key=config.openai_api_key)
        self.memory = MemoryKeeper(config.memory_db_path)
        
        # Initialize all agents
        self.agents = {
            'DataMiner': DataMinerAgent(self.llm_client, config),
            'Quant': QuantAgent(self.llm_client, config),
            'TrendScout': TrendScoutAgent(self.llm_client, config),
            'RiskAssessor': RiskAssessorAgent(self.llm_client, config),
            'ComplianceWatcher': ComplianceWatcherAgent(self.llm_client, config),
            'MarketPulse': MarketPulseAgent(self.llm_client, config)
        }
        
        logger.info(f"Initialized Cerebrum with {len(self.agents)} specialist agents")
    
    def analyze_request(self, request: str) -> Dict[str, Any]:
        """Determine which agents should handle the request"""
        prompt = f"""
        Analyze this financial analysis request and determine which specialists should be involved.
        
        Request: {request}
        
        Available specialists:
        - DataMiner: Document analysis, information extraction from texts
        - Quant: Numerical analysis, database queries, statistical calculations
        - TrendScout: Temporal patterns, trend analysis, seasonality detection
        - RiskAssessor: Risk identification, threat assessment, risk quantification
        - ComplianceWatcher: Regulatory compliance, legal requirements monitoring
        - MarketPulse: Market sentiment, real-time data, external factors
        
        Return a JSON response:
        {{
            "agents": ["agent1", "agent2"],  // List relevant agents
            "priority": "high|medium|low",
            "reasoning": "brief explanation"
        }}
        
        JSON:
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "agents": ["Quant", "TrendScout"],
                    "priority": "medium",
                    "reasoning": "Default agent selection"
                }
        except Exception as e:
            logger.warning(f"Request analysis failed: {e}")
            return {
                "agents": ["Quant", "TrendScout"],
                "priority": "medium",
                "reasoning": "Fallback agent selection"
            }
    
    def synthesize_findings(self, request: str, state: CerebrumState) -> str:
        """Combine insights from multiple agents into coherent analysis"""
        
        # Prepare findings summary
        findings_summary = []
        for finding in state.findings:
            agent_name = finding.get('agent', 'Unknown')
            insights = finding.get('insights', 'No insights provided')
            confidence = finding.get('confidence', 0)
            
            findings_summary.append(f"""
### {agent_name} Analysis (Confidence: {confidence:.1%})
{insights}
""")
        
        findings_text = "\n".join(findings_summary)
        
        # Prepare risk summary if risks identified
        risk_summary = ""
        if state.risks_identified:
            risk_summary = f"""
High-Priority Risks Identified: {len(state.risks_identified)}
""" + "\n".join([f"- {r.get('description', 'Unknown risk')}" for r in state.risks_identified[:5]])
        
        # Prepare compliance summary
        compliance_summary = ""
        if state.compliance_issues:
            compliance_summary = f"""
Compliance Issues Found: {len(state.compliance_issues)}
""" + "\n".join([f"- {c.get('description', 'Unknown issue')}" for c in state.compliance_issues[:5]])
        
        # Generate comprehensive synthesis
        synthesis_prompt = f"""
        You're a senior financial analyst synthesizing findings from multiple specialist teams.
        
        Original Request: {request}
        
        Team Findings:
        {findings_text}
        
        {risk_summary}
        
        {compliance_summary}
        
        Average Confidence: {state.get_average_confidence():.1%}
        
        Provide a comprehensive executive summary that:
        1. Directly answers the original request
        2. Synthesizes key findings across all analyses
        3. Highlights critical risks and opportunities
        4. Provides specific, actionable recommendations
        5. Identifies any conflicting findings and resolves them
        6. Concludes with next steps
        
        Format with clear sections and bullet points where appropriate.
        
        Executive Summary:
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.5,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "Failed to synthesize findings. Please review individual agent reports."
    
    def process_request(self, request: str, force_agents: List[str] = None) -> Dict[str, Any]:
        """Main processing pipeline for analysis requests"""
        logger.info(f"🎯 Processing request: {request[:100]}...")
        
        # Initialize state
        state = CerebrumState(request=request)
        
        # Step 1: Analyze request and plan approach
        if force_agents:
            selected_agents = force_agents
            priority = "high"
        else:
            plan = self.analyze_request(request)
            selected_agents = plan.get('agents', ['Quant', 'TrendScout'])
            priority = plan.get('priority', 'medium')
            state.metadata['plan_reasoning'] = plan.get('reasoning', '')
        
        # Limit number of agents if needed
        if len(selected_agents) > self.config.max_agents_per_request:
            selected_agents = selected_agents[:self.config.max_agents_per_request]
        
        logger.info(f"📋 Deploying agents: {', '.join(selected_agents)} (Priority: {priority})")
        state.plan = selected_agents
        
        # Step 2: Check memory for relevant insights
        relevant_memories = self.memory.recall_insights(request)
        if relevant_memories:
            logger.info(f"🧠 Found {len(relevant_memories)} relevant memories")
            state.metadata['memories_used'] = len(relevant_memories)
        
        # Step 3: Execute analysis with selected agents
        agent_results = []
        for agent_name in selected_agents:
            if agent_name in self.agents:
                try:
                    result = self.agents[agent_name].process(request, state)
                    agent_results.append(result)
                    logger.info(f"✅ {agent_name} completed (confidence: {result.get('confidence', 0):.1%})")
                except Exception as e:
                    logger.error(f"❌ {agent_name} failed: {e}")
                    agent_results.append({
                        'agent': agent_name,
                        'error': str(e),
                        'confidence': 0
                    })
        
        # Step 4: Synthesize findings
        final_analysis = self.synthesize_findings(request, state)
        state.final_report = final_analysis
        
        # Step 5: Store key insights for future reference
        insight_key = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a summary for memory
        memory_summary = {
            'request': request[:200],
            'agents_used': selected_agents,
            'average_confidence': state.get_average_confidence(),
            'key_findings': final_analysis[:500],
            'risks_count': len(state.risks_identified),
            'compliance_issues_count': len(state.compliance_issues)
        }
        
        self.memory.store_insight(
            insight_key,
            memory_summary,
            category='analysis',
            importance=state.get_average_confidence()
        )
        
        # Step 6: Clean up old memories periodically
        if len(self.memory.long_term) > 1000:
            self.memory.forget_old_memories()
        
        # Prepare final output
        output = {
            'request_id': insight_key,
            'request': request,
            'priority': priority,
            'agents_used': selected_agents,
            'individual_findings': agent_results,
            'final_analysis': final_analysis,
            'confidence_score': state.get_average_confidence(),
            'risks_identified': len(state.risks_identified),
            'compliance_issues': len(state.compliance_issues),
            'relevant_memories': [
                {'key': m['key'], 'relevance': m['relevance']} 
                for m in relevant_memories[:5]
            ],
            'processing_time': state.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✨ Analysis complete. Confidence: {output['confidence_score']:.1%}")
        
        return output
    
    def get_agent_specialties(self) -> Dict[str, str]:
        """Get a summary of all agent specialties"""
        return {name: agent.specialty for name, agent in self.agents.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health"""
        return {
            'agents_available': list(self.agents.keys()),
            'memory_stats': self.memory.get_memory_stats(),
            'config': {
                'max_agents_per_request': self.config.max_agents_per_request,
                'confidence_threshold': self.config.confidence_threshold,
                'proactive_monitoring': self.config.enable_proactive_monitoring
            },
            'system_health': 'operational'
        }


# ============================================================================
# Factory Function and Main Interface
# ============================================================================

def create_cerebrum_system(
    openai_api_key: str,
    database_path: str = "financials.db",
    vector_db_path: str = "./cerebrum_vectors",
    memory_db_path: str = "./cerebrum_memory.json"
) -> CerebrumOrchestrator:
    """
    Factory function to create a complete Cerebrum system.
    
    Args:
        openai_api_key: OpenAI API key for LLM access
        database_path: Path to SQLite database with financial data
        vector_db_path: Path to vector database for document storage
        memory_db_path: Path to persistent memory storage
    
    Returns:
        Configured CerebrumOrchestrator instance
    """
    
    config = CerebrumConfig(
        openai_api_key=openai_api_key,
        database_path=database_path,
        vector_db_path=vector_db_path,
        memory_db_path=memory_db_path
    )
    
    return CerebrumOrchestrator(config)


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Example of how to use the Cerebrum system"""
    
    # Initialize the system
    cerebrum = create_cerebrum_system(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        database_path="financials.db"
    )
    
    # Example queries
    example_queries = [
        "Analyze Microsoft's revenue growth trends and identify potential risks in the cloud computing segment",
        "What compliance requirements should we monitor for a fintech startup?",
        "Assess market sentiment for tech stocks and identify upcoming catalysts",
        "Compare Q3 performance against industry benchmarks and suggest improvements"
    ]
    
    # Process a query
    for query in example_queries[:1]:  # Process first query as example
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}\n")
        
        result = cerebrum.process_request(query)
        
        print("📊 CEREBRUM ANALYSIS COMPLETE")
        print(f"Confidence Score: {result['confidence_score']:.1%}")
        print(f"Agents Used: {', '.join(result['agents_used'])}")
        print(f"Risks Identified: {result['risks_identified']}")
        print(f"Compliance Issues: {result['compliance_issues']}")
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY:")
        print("="*80)
        print(result['final_analysis'])
        
    # Display system status
    print("\n" + "="*80)
    print("SYSTEM STATUS:")
    print("="*80)
    status = cerebrum.get_system_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run example
    example_usage()
