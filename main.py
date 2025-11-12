from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevance
from ragas import evaluate

def run_cricket_evaluation():
    data = {
        "question": ["Who won the 2011 Cricket World Cup?"],
        "answer": ["India won the 2011 Cricket World Cup by defeating Sri Lanka in the final."],
        "contexts": [["The 2011 ICC Cricket World Cup final was played between India and Sri Lanka. India won by 6 wickets."]],
        "ground_truth": ["India won the 2011 Cricket World Cup final against Sri Lanka in Mumbai."]
    }

    dataset = Dataset.from_dict(data)

    results = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevance])

    print("\n Cricket Answer Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    print("=== Evaluating Cricket Question Answer ===")
    run_cricket_evaluation()
    print("\n Done! Now try comparing prompts with Promptfoo.")
