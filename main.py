# from fastapi import fastapi
# from api import router

# app = fastapi()
# app.include_router(router)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.routers import api  # Assuming you have this in your project
app = FastAPI()
 
# Configure CORS
origins = [
     "*"  # Allows all origins (use with caution in production)
]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific domains or use ["*"] for all origins
    allow_credentials=True,  # Whether to allow cookies to be sent with requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
 
# Include the router
app.include_router(api.router)
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)