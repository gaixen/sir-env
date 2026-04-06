from openenv_service import app
import uvicorn


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
