from fastapi import FastAPI
from routes import wrapper

app = FastAPI()
app.include_router(wrapper.router)
