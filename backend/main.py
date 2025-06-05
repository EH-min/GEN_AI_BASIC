from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from domain import ai_router

app = FastAPI()

origins = [
    "http://localhost:5000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(ai_router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        reload=True,
        host="localhost",
        port=8000
    )