from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class MatchRequest(BaseModel):
    skillReq: str
    skills: List[str]

@app.post("/match")
async def match_skills(data: MatchRequest):
    skillReq_vec = model.encode(data.skillReq)
    skills_vecs = model.encode(data.skills)

    results = []
    for i, skills_vecs in enumerate(skills_vecs):
        sim = cosine_similarity([skillReq_vec], [skills_vecs])[0][0]
        results.append({
            'skill_index': i,
            'skill_text': data.skills[i],
            'similarity': float(round(sim, 4))
        })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
