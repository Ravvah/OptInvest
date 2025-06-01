from fastapi import FastAPI

from app.api.routers import health, simulation, prediction

app = FastAPI(title="OPTINVEST API", version="0.1.0")

#endpoints
app.include_router(health.router)
app.include_router(simulation.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host= "0.0.0.0", port=8000)

