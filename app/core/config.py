from pydantic import BaseSettings


class Settings(BaseSettings):
    model_weights_path: str = "/models/model_weights.pth"
    device: str = "cuda"

    class Config:
        env_file = ".env"


settings = Settings()
