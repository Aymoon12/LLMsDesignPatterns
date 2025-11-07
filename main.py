from datetime import datetime

from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import logging
import ssl
import urllib3
import httpx
import anthropic
import time
import re
import json
from google import genai
from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main")


# Load environment variables from .env file
load_dotenv()


# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Patch the Client request method to disable SSL verification
_original_request = httpx.Client.request

def _patched_request(self, *args, **kwargs):
    # Force verify=False for all requests
    if hasattr(self, '_transport'):
        self._transport._pool._ssl_context = ssl._create_unverified_context()
    return _original_request(self, *args, **kwargs)

httpx.Client.request = _patched_request


# Get Supabase credentials from environment variables and create the client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# Patterns from the Gang of Four design patterns for fallback methods
GOF_PATTERNS = [
    'Singleton', 'Factory Method', 'Abstract Factory', 'Builder', 'Prototype',
    'Adapter', 'Bridge', 'Composite', 'Decorator', 'Facade', 'Flyweight', 'Proxy',
    'Chain of Responsibility', 'Command', 'Interpreter', 'Iterator', 'Mediator',
    'Memento', 'Observer', 'State', 'Strategy', 'Template Method', 'Visitor'
]


class SupabaseLoader:

    @staticmethod
    def load_all_requirements() -> List[Dict]:
        """Load all requirements with their ground truth patterns"""
        try:
            # Get all requirements
            response = supabase.table("requirements").select("*").execute()
            requirements = []

            for row in response.data:
                # Get ground truth patterns for this requirement
                gt_response = supabase.table("requirement_ground_truth") \
                    .select("pattern_name, is_primary") \
                    .eq("requirement_id", row["id"]) \
                    .order("is_primary", desc=True) \
                    .execute()

                patterns = [p["pattern_name"] for p in gt_response.data]
                primary_patterns = [p["pattern_name"] for p in gt_response.data if p["is_primary"]]

                requirements.append({
                    "id": row["id"],
                    "text": row["requirement_type"],
                    "source_type": row["source_type"],
                    "ground_truth_patterns": patterns,
                    "primary_patterns": primary_patterns,
                    "metadata": row.get("metadata", {}),
                    "notes": row.get("notes", "")
                })

            logger.info(f"Loaded {len(requirements)} requirements from Supabase")
            return requirements

        except Exception as e:
            logger.error(f"Error loading requirements: {e}")
            return []

    @staticmethod
    def load_specific_rows(min: int, max: int) -> List[Dict]:
        """Load all requirements with their ground truth patterns"""
        try:
            # Get all requirements
            response = (supabase.table("requirements")
                        .select("*")
                        .gte("id", min)
                        .lte("id", max)
                        .execute())
            requirements = []

            for row in response.data:
                # Get ground truth patterns for this requirement
                gt_response = supabase.table("requirement_ground_truth") \
                    .select("pattern_name, is_primary") \
                    .eq("requirement_id", row["id"]) \
                    .order("is_primary", desc=True) \
                    .execute()

                patterns = [p["pattern_name"] for p in gt_response.data]
                primary_patterns = [p["pattern_name"] for p in gt_response.data if p["is_primary"]]

                requirements.append({
                    "id": row["id"],
                    "text": row["requirement_type"],
                    "source_type": row["source_type"],
                    "ground_truth_patterns": patterns,
                    "primary_patterns": primary_patterns,
                    "metadata": row.get("metadata", {}),
                    "notes": row.get("notes", "")
                })

            logger.info(f"Loaded {len(requirements)} requirements from Supabase")
            return requirements

        except Exception as e:
            logger.error(f"Error loading requirements: {e}")
            return []



    @staticmethod
    def load_by_source(source_type: str) -> List[Dict]:
        response = supabase.table("requirements") \
            .select("*") \
            .eq("source_type", source_type) \
            .execute()

        requirements = []
        for row in response.data:
            gt_response = supabase.table("requirement_ground_truth") \
                .select("pattern_name") \
                .eq("requirement_id", row["id"]) \
                .execute()

            patterns = [p["pattern_name"] for p in gt_response.data]
            requirements.append({
                "id": row["id"],
                "text": row["requirement_type"],
                "ground_truth_patterns": patterns,
                "source_type": row["source_type"],
            })

        logger.info(f"Loaded {len(requirements)} requirements from Supabase")

        return requirements


class PromptGenerator:

    @staticmethod
    def zero_shot_prompt(requirement_text: str) -> str:
        """No examples prompt"""
        return f"""You are a software design expert. Given the following software requirement, identify the most appropriate Gang of Four (GoF) design pattern(s) that should be used.
        Requirement: {requirement_text}
        
        Provide your response in the following JSON format:
        {{
            "patterns": ["Pattern Name 1", "Pattern Name 2"],
            "confidence": {{"Pattern Name 1": 0.95, "Pattern Name 2": 0.85}},
            "reasoning": "Explain why you chose these patterns and how they address the requirement"
        }}
        
        Only recommend patterns from the 23 Gang of Four design patterns. If multiple patterns could apply, list them in order of appropriateness."""

    @staticmethod
    def few_shot_prompt(requirement_text: str) -> str:
        """With examples prompt"""
        return f"""You are a software design expert. Given a software requirement, identify the most appropriate Gang of Four (GoF) design pattern(s).

        Example 1:
        Requirement: "The system needs to ensure only one database connection pool exists"
        Response: {{"patterns": ["Singleton"], "confidence": {{"Singleton": 0.98}}, "reasoning": "This requirement explicitly needs a single instance with global access, which is the core intent of the Singleton pattern."}}
        
        Example 2:
        Requirement: "Different payment methods should be selectable at runtime without changing client code"
        Response: {{"patterns": ["Strategy"], "confidence": {{"Strategy": 0.95}}, "reasoning": "The requirement describes interchangeable algorithms (payment methods) that can be selected at runtime, which is addressed by the Strategy pattern."}}
        
        Now analyze this requirement:
        Requirement: {requirement_text}
        
        Provide your response in JSON format with patterns, confidence scores, and reasoning."""


class LLMEvaluator:
    """Handles interaction with multiple LLM APIs"""

    def __init__(self):
        self.anthropic_client = None
        self.gemini_client = None
        self.xai_client = None
        self.openai_client = None
        self.deepseek_client = None
        self.setup_clients()


    def setup_clients(self):

        http_client = httpx.Client(verify=False, timeout=60.0)
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("CLAUDE_API_KEY"),
            http_client=http_client)
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.xai_client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
            http_client=http_client
        )
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI()

        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            http_client=http_client
        )


    def query_claude(self, prompt: str, temperature: float = 0.3, max_retries: int = 3) -> tuple[Any, int] | tuple[
        None, int] | None:
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                message = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_time = int((time.time() - start_time) * 1000)
                return message.content[0].text, response_time

            except KeyboardInterrupt:
                logger.warning("Query interrupted by user (Ctrl+C)")
                raise

            except Exception as e:
                logger.warning(f"Error querying Claude (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to query Claude after {max_retries} attempts")
                    return None, 0
        return None

    def query_gemini(self, prompt: str, temperature: float = 0.3) -> tuple[Any, int] | tuple[None, int]:
        """Query Gemini via Google AI API"""
        try:
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                 config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1000
                )
            )
            response_time = int((time.time() - start_time) * 1000)

            return response.text, response_time
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return None, 0

    def query_grok(self, prompt: str, temperature: float = 0.3) -> tuple[Any, int] | tuple[None, int]:
        """Query Grok via XAI API"""

        try:
            start = time.time()

            system_prompt = "You are Grok, a highly intelligent, helpful AI assistant."

            response = self.xai_client.chat.completions.create(
                model="grok-4-fast",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000,
            )
            response_time = int((time.time() - start) * 1000)

            return response.choices[0].message.content, response_time
        except Exception as e:
            print(f"Error querying Grok: {e}")
            return None, 0

    def query_gpt5(self, prompt: str, temperature: float = 0.3) -> tuple[Any, int] | tuple[None, int]:
        """Query OpenAI GPT-5"""
        try:
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            response_time = int((time.time() - start_time) * 1000)
            response_text = response.choices[0].message.content
            return response_text, response_time
        except Exception as e:
            print(f"Error querying OpenAI GPT-5: {e}")
            return None, 0

    def query_deepseek(self, prompt: str, temperature: float = 0.3) -> tuple[Any, int] | tuple[None, int]:
        """Query DeepSeek"""
        try:
            start_time = time.time()
            system_prompt = "You are DeepSeek, a highly intelligent, helpful AI assistant."

            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            response_time = int((time.time() - start_time) * 1000)
            response_text = response.choices[0].message.content
            return response_text, response_time
        except Exception as e:
            print(f"Error querying Deepseek: {e}")
            return None, 0



    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response to extract patterns, confidence, and reasoning"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'patterns': data.get('patterns', []),
                    'confidence': data.get('confidence', {}),
                    'reasoning': data.get('reasoning', data.get('analysis', ''))
                }
        except json.JSONDecodeError:
            pass

        # Fallback: extract pattern names mentioned in response
        patterns = [p for p in GOF_PATTERNS if p in response]
        return {
            'patterns': patterns[:3],
            'confidence': {p: 0.5 for p in patterns[:3]},
            'reasoning': response
        }

    def evaluate_requirement(
            self,
            requirement: Dict,
            model_name: str,
            prompt_type: str = 'zero-shot',
            temperature: float = 0.3
    ) -> Optional[Dict]:

        # Generate appropriate prompt
        if prompt_type == 'zero-shot':
            prompt = PromptGenerator.zero_shot_prompt(requirement['text'])
        elif prompt_type == 'few-shot':
            prompt = PromptGenerator.few_shot_prompt(requirement['text'])
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Query appropriate model
        if model_name == 'claude':
            raw_response, response_time = self.query_claude(prompt, temperature)
        elif model_name == 'gemini':
            raw_response, response_time = self.query_gemini(prompt, temperature)
        elif model_name == 'grok':
            raw_response, response_time = self.query_grok(prompt, temperature)
        elif model_name == 'gpt5':
            raw_response, response_time = self.query_gpt5(prompt, temperature)
        elif model_name == 'deepseek':
            raw_response, response_time = self.query_deepseek(prompt, temperature)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if raw_response is None:
            return None

        # Parse response
        parsed = self._parse_response(raw_response)

        return {
            'requirement_id': requirement['id'],
            'model_name': model_name,
            'prompt_type': prompt_type,
            'temperature': temperature,
            'raw_response': raw_response,
            'reasoning': parsed['reasoning'],
            'predicted_patterns': parsed['patterns'],
            'confidence_scores': parsed['confidence'],
            'response_time_ms': response_time,
            'timestamp': datetime.now().isoformat()
        }

    def save_response_to_supabase(self, response: Dict):
        try:
            if response is not None:
                supabase.table("model_responses").insert(response).execute()
        except Exception as e:
            logger.error(f"Error saving response: {e}")


class QuantitativeAnalyzer:
    """Compute quantitative metrics from Supabase data"""

    def __init__(self):
        self.load_data()

    def load_data(self):
        """Load requirements and responses from Supabase"""
        # Load requirements with ground truth
        req_response = supabase.table("requirements").select("*").execute()
        self.requirements = {}

        for row in req_response.data:
            gt_response = supabase.table("requirement_ground_truth") \
                .select("pattern_name, is_primary") \
                .eq("requirement_id", row["id"]) \
                .execute()

            patterns = [p["pattern_name"] for p in gt_response.data]
            primary_patterns = [p["pattern_name"] for p in gt_response.data if p["is_primary"]]

            self.requirements[row["id"]] = {
                "id": row["id"],
                "text": row["requirement_type"],
                "source_type": row["source_type"],
                "ground_truth": patterns,
                "primary_patterns": primary_patterns
            }

        # Load model responses
        resp_response = supabase.table("model_responses").select("*").execute()
        self.responses = resp_response.data

        # Build analysis DataFrame
        self._build_dataframe()

        print(f"Loaded {len(self.requirements)} requirements")
        print(f"Loaded {len(self.responses)} model responses")

    def _build_dataframe(self):
        """Convert responses to pandas DataFrame for analysis"""
        data = []

        for resp in self.responses:
            req = self.requirements[resp["requirement_id"]]

            predicted = set(resp["predicted_patterns"])
            ground_truth = set(req["ground_truth"])
            primary_gt = set(req["primary_patterns"])

            # Calculate correctness metrics
            exact_match = predicted == ground_truth
            partial_match = len(predicted & ground_truth) > 0
            top1_correct = (len(resp["predicted_patterns"]) > 0 and
                            resp["predicted_patterns"][0] in req["ground_truth"])
            primary_correct = (len(resp["predicted_patterns"]) > 0 and
                               resp["predicted_patterns"][0] in req["primary_patterns"])

            confidence_scores = resp.get("confidence_scores") or {}
            avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0

            data.append({
                'requirement_id': resp["requirement_id"],
                'model': resp["model_name"],
                'prompt_type': resp["prompt_type"],
                'source_type': req["source_type"],
                'ground_truth': list(ground_truth),
                'predicted': list(predicted),
                'exact_match': exact_match,
                'partial_match': partial_match,
                'top1_correct': top1_correct,
                'primary_correct': primary_correct,
                'num_patterns_predicted': len(predicted),
                'confidence_avg': avg_confidence,
                'reasoning_length': len(resp.get("reasoning", "")),
                'response_time_ms': resp.get("response_time_ms", 0)
            })

        self.df = pd.DataFrame(data)

    def compute_overall_metrics(self) -> pd.DataFrame:
        """Compute overall accuracy, precision, recall"""
        metrics = []

        for model in self.df['model'].unique():
            for prompt_type in self.df['prompt_type'].unique():
                subset = self.df[
                    (self.df['model'] == model) &
                    (self.df['prompt_type'] == prompt_type)
                    ]

                if len(subset) == 0:
                    continue

                metrics.append({
                    'model': model,
                    'prompt_type': prompt_type,
                    'count': len(subset),
                    'exact_match_accuracy': subset['exact_match'].mean(),
                    'partial_match_accuracy': subset['partial_match'].mean(),
                    'top1_accuracy': subset['top1_correct'].mean(),
                    'primary_pattern_accuracy': subset['primary_correct'].mean(),
                    'avg_confidence': subset['confidence_avg'].mean(),
                    'avg_patterns_predicted': subset['num_patterns_predicted'].mean(),
                    'avg_response_time_ms': subset['response_time_ms'].mean()
                })

        return pd.DataFrame(metrics)

    def compute_per_pattern_metrics(self) -> pd.DataFrame:
        """Compute precision, recall, F1 for each pattern"""
        results = []

        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]

            for pattern in GOF_PATTERNS:
                tp = sum(1 for _, row in model_df.iterrows()
                         if pattern in row['ground_truth'] and pattern in row['predicted'])
                fp = sum(1 for _, row in model_df.iterrows()
                         if pattern not in row['ground_truth'] and pattern in row['predicted'])
                fn = sum(1 for _, row in model_df.iterrows()
                         if pattern in row['ground_truth'] and pattern not in row['predicted'])

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results.append({
                    'model': model,
                    'pattern': pattern,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': tp + fn
                })

        return pd.DataFrame(results)

    def compute_stratified_metrics(self, stratify_by: str) -> pd.DataFrame:
        """Compute metrics stratified by source_type"""
        results = []

        for model in self.df['model'].unique():
            for category in self.df[stratify_by].unique():
                subset = self.df[
                    (self.df['model'] == model) &
                    (self.df[stratify_by] == category)
                    ]

                if len(subset) == 0:
                    continue

                results.append({
                    'model': model,
                    stratify_by: category,
                    'exact_match_accuracy': subset['exact_match'].mean(),
                    'top1_accuracy': subset['top1_correct'].mean(),
                    'count': len(subset)
                })

        return pd.DataFrame(results)

    def compute_confusion_matrix(self, model_name: str) -> pd.DataFrame:
        """Generate confusion matrix for a specific model"""
        model_df = self.df[self.df['model'] == model_name]

        y_true = []
        y_pred = []

        for _, row in model_df.iterrows():
            if len(row['ground_truth']) > 0 and len(row['predicted']) > 0:
                y_true.append(row['ground_truth'][0])
                y_pred.append(row['predicted'][0])

        if len(y_true) == 0:
            return pd.DataFrame()

        cm = confusion_matrix(y_true, y_pred, labels=GOF_PATTERNS)
        return pd.DataFrame(cm, index=GOF_PATTERNS, columns=GOF_PATTERNS)

    def export_metrics_report(self, output_dir: str = "./results"):
        """Export comprehensive metrics report"""
        os.makedirs(output_dir, exist_ok=True)

        # Overall metrics
        overall = self.compute_overall_metrics()
        overall.to_csv(f"{output_dir}/overall_metrics.csv", index=False)
        print(f"Saved overall metrics")
        print(overall.to_string())

        # Per-pattern metrics
        per_pattern = self.compute_per_pattern_metrics()
        per_pattern.to_csv(f"{output_dir}/per_pattern_metrics.csv", index=False)
        print(f"\nSaved per-pattern metrics")

        # Stratified by source
        by_source = self.compute_stratified_metrics('source_type')
        by_source.to_csv(f"{output_dir}/metrics_by_source.csv", index=False)
        print(f"Saved metrics by source type")
        print(by_source.to_string())

        print(f"\nAll metrics exported to {output_dir}/")

def main():

    loader = SupabaseLoader()
    # requirements = loader.load_specific_rows(48, 72)
    #
    # if not requirements:
    #     logger.error("No requirements found in Supabase")
    #     return
    #
    # evaluator = LLMEvaluator()
    # model_names= ['claude', 'gemini', 'grok','deepseek']
    #
    # for model_name in model_names:
    #     for req in requirements:
    #         evaluation = evaluator.evaluate_requirement(req, model_name, 'zero-shot', 0.3)
    #         evaluator.save_response_to_supabase(evaluation)

    analyzer = QuantitativeAnalyzer()

    if len(analyzer.responses) == 0:
        print("No responses found in database. Run evaluations first.")
        return

    # Export all metrics
    analyzer.export_metrics_report(output_dir="./results")


if __name__ == "__main__":
    main()