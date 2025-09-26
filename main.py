from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class SkillsRequest(BaseModel):
    skills: List[str]

@app.post("/match")
def match_skills(request: SkillsRequest):
    embeddings = model.encode(request.skills)

    similarity = cosine_similarity([embeddings[0]], embeddings[1:])

    return {"similarity_scores": similarity.tolist()}
