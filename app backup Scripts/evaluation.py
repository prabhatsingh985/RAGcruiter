# evaluation.py
from typing import List
from evaluate import load
from sentence_transformers import SentenceTransformer, util

bleu_metric = load("bleu")
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_bleu_score(generated: str, reference: str) -> float:
    result = bleu_metric.compute(predictions=[generated], references=[[reference]])
    return result["bleu"]

def embedding_similarity(text1: str, text2: str) -> float:
    embeddings = model.encode([text1, text2])
    return float(util.cos_sim(embeddings[0], embeddings[1])[0])

def evaluate_skills(extracted_skills: List[str], required_skills: List[str]) -> dict:
    extracted = set(skill.lower() for skill in extracted_skills)
    required = set(skill.lower() for skill in required_skills)

    true_positives = extracted & required
    precision = len(true_positives) / len(extracted) if extracted else 0
    recall = len(true_positives) / len(required) if required else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "matched_skills": list(true_positives)
    }
