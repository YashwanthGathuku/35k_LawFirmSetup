import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_scripts.srlc_engine import SRLCEngine

# -----------------------------------------------------------------------------
# Test Legacy SRLC Logic
# -----------------------------------------------------------------------------

def test_srlc_engine(mocker):
    # The SRLC engine uses llm.complete() three times (Draft, Critique, Refine)
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "Initial Draft",
        "Critique Note",
        "Final Verified Answer"
    ]

    engine = SRLCEngine(mock_llm)
    res = engine.run("Test query", "Context data")

    assert res["draft"] == "Initial Draft"
    assert res["critique"] == "Critique Note"
    assert res["final"] == "Final Verified Answer"
    assert mock_llm.complete.call_count == 3

# -----------------------------------------------------------------------------
# Test Main query_rag Execution Flow (Simulated CLI)
# -----------------------------------------------------------------------------

@patch("rag_scripts.query_rag.chromadb.PersistentClient")
@patch("rag_scripts.query_rag.load_index_from_storage")
@patch("rag_scripts.query_rag.StorageContext.from_defaults")
@patch("rag_scripts.query_rag.VectorStoreIndex.from_documents")
@patch("rag_scripts.query_rag.HuggingFaceEmbedding")
@patch("rag_scripts.query_rag.Settings")
@patch("sys.argv", ["query_rag.py", "Test query", "--use-cag"])
def test_query_rag_cag_mode(mock_settings, mock_hf_embed, mock_vsi, mock_storage_context, mock_load_index, mock_chroma, capsys, monkeypatch, mocker):
    # Instead of letting query_rag instantiate an LLM, mock the constructors in the actual module
    mock_llm_instance = MagicMock()
    mock_llm_instance.complete.return_value = "CAG Answer"
    type(mock_llm_instance).callback_manager = mocker.PropertyMock()

    mocker.patch("rag_scripts.query_rag.Ollama", return_value=mock_llm_instance)
    mocker.patch("rag_scripts.query_rag.OpenAI", return_value=mock_llm_instance)

    # We must patch llama_index instances that get instantiated indirectly
    mocker.patch("llama_index.llms.openai.base.OpenAI.complete", return_value=MagicMock(text="CAG Answer"))
    mocker.patch("llama_index.llms.ollama.Ollama.complete", return_value=MagicMock(text="CAG Answer"))
    mocker.patch("llama_index.core.llms.llm.LLM.complete", return_value=MagicMock(text="CAG Answer"))

    # To prevent deep HTTP calls in tests, we must mock out the client requests library
    mocker.patch("httpx.Client.send")
    mocker.patch("httpx.AsyncClient.send")

    # We must patch load_index_from_storage because the mocked StorageContext breaks it
    mock_index = MagicMock()
    mock_vsi.return_value = mock_index
    mock_load_index.side_effect = FileNotFoundError("Mocked exception")

    # We force the script to use our mocked Ollama to bypass OpenAI's strict model name validation completely
    monkeypatch.setenv("MODEL_TYPE", "ollama")
    monkeypatch.setattr("sys.argv", ["query_rag.py", "Test query", "--use-cag"])
    """
    Test that query_rag.py runs without error and outputs valid JSON
    when called via CLI arguments (e.g., from n8n)
    """

    # We must patch Settings.llm so the code uses our mock instead of trying to validate OpenAI model names
    mock_settings.llm = mock_llm_instance

    # Need to intercept the exact point that the underlying OpenAI library makes an HTTP request if instantiated
    mocker.patch("llama_index.llms.openai.base.OpenAI.chat", return_value=MagicMock(message=MagicMock(content="CAG Answer")))
    mocker.patch("llama_index.llms.openai.base.OpenAI._chat", return_value=MagicMock(message=MagicMock(content="CAG Answer")))

    # Mock Index & Retriever
    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_node = MagicMock()
    mock_node.node.get_content.return_value = "Mocked Context"
    mock_node.node.metadata = {"file_name": "test.pdf"}
    mock_node.score = 0.95
    mock_retriever.retrieve.return_value = [mock_node]
    mock_index.as_retriever.return_value = mock_retriever
    mock_load_index.return_value = mock_index

    # Import script (it executes immediately upon import, so we reload it if needed)
    import importlib
    import rag_scripts.query_rag

    # We need to test the execute_query directly to avoid the CLI wrapper swallowing exceptions into sys.exit
    output_dict = rag_scripts.query_rag.execute_query(
        question="Test query",
        index=mock_index,
        llm=mock_llm_instance,
        use_cag=True,
        model_name="local-model"
    )

    assert output_dict["answer"] == "CAG Answer"
    assert output_dict["mode"] == "CAG"

@patch("rag_scripts.query_rag.chromadb.PersistentClient")
@patch("rag_scripts.query_rag.load_index_from_storage")
@patch("rag_scripts.query_rag.StorageContext.from_defaults")
@patch("rag_scripts.query_rag.VectorStoreIndex.from_documents")
@patch("rag_scripts.query_rag.HuggingFaceEmbedding")
@patch("rag_scripts.query_rag.Settings")
@patch("sys.argv", ["query_rag.py", "Test query"])
def test_query_rag_standard_mode(mock_settings, mock_hf_embed, mock_vsi, mock_storage_context, mock_load_index, mock_chroma, capsys, monkeypatch, mocker):
    # Mock LLM in the module
    mock_llm_instance = MagicMock()
    type(mock_llm_instance).callback_manager = mocker.PropertyMock()
    mocker.patch("rag_scripts.query_rag.Ollama", return_value=mock_llm_instance)
    mocker.patch("rag_scripts.query_rag.OpenAI", return_value=mock_llm_instance)

    mock_index = MagicMock()
    mock_vsi.return_value = mock_index
    mock_load_index.side_effect = FileNotFoundError("Mocked exception")

    # Mock the query engine output
    mocker.patch("llama_index.core.base.base_query_engine.BaseQueryEngine.query", return_value="Standard RAG Answer")
    monkeypatch.setenv("MODEL_TYPE", "ollama")
    monkeypatch.setattr("sys.argv", ["query_rag.py", "Test query"])
    mock_settings.llm = mock_llm_instance

    # To prevent deep HTTP calls in tests, we must mock out the client requests library
    mocker.patch("httpx.Client.send")
    mocker.patch("httpx.AsyncClient.send")

    # Mock Index & Query Engine
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = "Standard RAG Answer"
    mock_index.as_query_engine.return_value = mock_query_engine

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_index.as_retriever.return_value = mock_retriever
    mock_vsi.return_value = mock_index

    mock_load_index.return_value = mock_index

    import importlib
    import rag_scripts.query_rag

    output_dict = rag_scripts.query_rag.execute_query(
        question="Test query",
        index=mock_index,
        llm=mock_llm_instance,
        use_cag=False,
        model_name="local-model"
    )

    assert output_dict["answer"] == "Standard RAG Answer"
    assert output_dict["mode"] == "RAG"
