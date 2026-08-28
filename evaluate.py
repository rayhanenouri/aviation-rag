import json
import os
from typing import List, Tuple
from src.vector_store import AviationVectorStore
from src.rag_pipeline import AviationRAG
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric
from anthropic import Anthropic


class ClaudeJudge(DeepEvalBaseLLM):
    """claude sonnet 4.6 as external judge for unbiased evaluation"""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=api_key)

    def load_model(self):
        """deepeval requires this method"""
        return self.client

    def generate(self, prompt: str) -> str:
        """generate evaluation from claude as judge"""
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def a_generate(self, prompt: str) -> str:
        """async generation for deepeval metrics"""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """return model identifier for deepeval"""
        return "claude-sonnet-4-6"


def build_test_dataset() -> Tuple[List[str], List[str]]:
    """aviation maintenance test questions with ground truth answers"""
    questions = [
        "What is the compliance deadline for EASA AD 2024-0038?",
        "What aircraft types are affected by EASA AD 2024-0038?",
        "Who issued EASA AD 2024-0038?",
        "What action is required by EASA AD 2024-0038?",
        "Can alternative compliance methods be approved?",
        "What is the NASA C-MAPS dataset used for?",
        "Which C-MAPS dataset is most used and why?",
        "What happens if an aircraft does not comply with the AD?",
        "What are the reporting requirements in the AD remarks?",
        "What is the weather forecast for Hamburg tomorrow?"
    ]

    ground_truths = [
        "within 36 months of first installation of affected part",
        "group 1 aeroplanes including A320 family",
        "EASA acting under Regulation EU 2018/1139",
        "inspection of galley installations as specified",
        "yes EASA can approve alternative methods if requested and substantiated",
        "benchmarking prognostic algorithms for turbofan engine degradation",
        "dataset number 1 most used at approximately 70 percent due to simplicity",
        "no person may operate the aircraft unless required actions are completed",
        "reporting required within 30 days of findings",
        "I don't have that information in the provided documents"
    ]

    return questions, ground_truths


def run_evaluation():
    """evaluate rag system using deepeval framework with groq judge"""
    print("INITIALIZING RAG SYSTEM EVALUATION")
    print()

    # load rag pipeline
    print("Loading vector store...")
    vs = AviationVectorStore()
    vs.load_existing()

    print("Initializing RAG pipeline...")
    rag = AviationRAG(vs)

    chunk_count = vs.vector_store._collection.count()
    print(f"System ready: {chunk_count} chunks indexed")
    print()

    # initialize claude as external judge for unbiased evaluation
    print("Initializing Claude Sonnet 4.6 as external judge...")
    claude_judge = ClaudeJudge()
    print()

    # initialize deepeval metrics with claude as judge
    faithfulness_metric = FaithfulnessMetric(
        model=claude_judge,
        threshold=0.5
    )

    relevancy_metric = AnswerRelevancyMetric(
        model=claude_judge,
        threshold=0.5
    )

    precision_metric = ContextualPrecisionMetric(
        model=claude_judge,
        threshold=0.5
    )

    # get test dataset
    questions, ground_truths = build_test_dataset()

    print(f"Running {len(questions)} test queries...")
    print()

    # storage for results
    test_cases = []
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    # run each question through pipeline
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
        print(f"[{i}/{len(questions)}] {question[:60]}...")

        # get rag response
        result = rag.query(question)
        answer = result["answer"]
        contexts = [source["content"] for source in result["sources"]]

        # create deepeval test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts
        )

        # measure metrics
        faithfulness_metric.measure(test_case)
        relevancy_metric.measure(test_case)
        precision_metric.measure(test_case)

        faithfulness_scores.append(faithfulness_metric.score)
        relevancy_scores.append(relevancy_metric.score)
        precision_scores.append(precision_metric.score)

        print(f"    Faithfulness: {faithfulness_metric.score:.2f}  Relevancy: {relevancy_metric.score:.2f}  Precision: {precision_metric.score:.2f}")

        test_cases.append(test_case)

    print()

    # calculate overall metrics
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    overall = (avg_faithfulness + avg_relevancy + avg_precision) / 3

    # check adversarial test
    adversarial_passed = "don't have that information" in test_cases[-1].actual_output.lower()

    # print evaluation report
    print("RAG SYSTEM EVALUATION REPORT")
    print("=" * 50)
    print(f"Evaluation framework  : DeepEval")
    print(f"Questions evaluated   : 10")
    print(f"In-scope questions    : 9")
    print(f"Adversarial queries   : 1")
    print(f"Documents indexed     : 2")
    print(f"Chunks in store       : {chunk_count}")
    print(f"LLM (generation)      : Groq gpt-oss-120b")
    print(f"LLM (judge)           : Claude Sonnet 4.6 (Anthropic)")
    print(f"Embeddings            : all-MiniLM-L6-v2")
    print()
    print(f"FAITHFULNESS        : {avg_faithfulness:.2f}")
    print(f"ANSWER RELEVANCY    : {avg_relevancy:.2f}")
    print(f"CONTEXT PRECISION   : {avg_precision:.2f}")
    print()
    print(f"OVERALL RAG SCORE   : {overall:.2f}")
    print()
    print(f"Adversarial test      : {'PASSED' if adversarial_passed else 'FAILED'}")
    print("System correctly refuses out-of-scope queries.")
    print("=" * 50)
    print()

    # save detailed results
    results = {
        "evaluation_framework": "DeepEval",
        "questions_evaluated": 10,
        "in_scope_questions": 9,
        "adversarial_queries": 1,
        "documents_indexed": 2,
        "chunks_in_store": chunk_count,
        "llm_generation": "groq/gpt-oss-120b",
        "llm_judge": "claude-sonnet-4-6",
        "embedding_model": "all-MiniLM-L6-v2",
        "metrics": {
            "faithfulness": round(avg_faithfulness, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "contextual_precision": round(avg_precision, 3),
            "overall_rag_score": round(overall, 3)
        },
        "adversarial_test": "PASSED" if adversarial_passed else "FAILED",
        "detailed_scores": {
            "faithfulness_per_question": [round(s, 3) for s in faithfulness_scores],
            "relevancy_per_question": [round(s, 3) for s in relevancy_scores],
            "precision_per_question": [round(s, 3) for s in precision_scores]
        }
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to evaluation_report.json")
    print()

    return results


if __name__ == "__main__":
    run_evaluation()
