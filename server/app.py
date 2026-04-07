from openenv_service import app
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    _PORT = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=_PORT)


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
