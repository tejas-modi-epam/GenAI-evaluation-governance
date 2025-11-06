from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevance
from ragas import evaluate

def run_ragas_module():
    data = {
        "question": ["What is RAG?"],
        "answer": ["RAG stands for Retrieval-Augmented Generation."],
        "contexts": [["RAG is a method that retrieves documents before answering."]],
        "ground_truth": ["RAG combines retrieval and generation to answer questions."]
    }

    dataset = Dataset.from_dict(data)

    results = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevance])

    print("\n RAGAS Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    print("=== Running Simple RAG Evaluation ===")
    run_ragas_module()
    print("\n Done! You can now test prompts with Promptfoo (next step).")
