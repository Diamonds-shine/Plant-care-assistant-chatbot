import pytest

from app import create_app
from nlp.chatbot import PlantCareChatbot


@pytest.fixture(scope="session")
def bot():
    """Build the TF-IDF index once for the whole test run."""
    return PlantCareChatbot()


@pytest.fixture()
def client(bot):
    app = create_app(chatbot=bot)
    app.config["TESTING"] = True
    return app.test_client()
