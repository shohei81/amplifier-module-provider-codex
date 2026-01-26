
import json
import tempfile
from pathlib import Path
from datetime import datetime

from amplifier_module_provider_codex.sessions import SessionManager
from amplifier_module_provider_codex.sessions import SessionState


def test_session_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(session_dir=tmpdir)
        
        # Create a session
        session = manager.create_session(name="test-session")
        session.metadata.total_tokens = 100
        session.add_message("user", "Hello")
        
        # Save it
        path = manager.save_session(session)
        assert path.exists()
        
        # Verify JSON content directly to ensure format is as expected
        with open(path) as f:
            data = json.load(f)
            assert data["metadata"]["name"] == "test-session"
            # Pydantic V2 serializes datetime as ISO string by default
            assert isinstance(data["metadata"]["created_at"], str)
            assert "T" in data["metadata"]["created_at"]
            
        # Load it back
        loaded_session = manager.load_session(session.metadata.session_id)
        assert loaded_session is not None
        assert loaded_session.metadata.session_id == session.metadata.session_id
        assert loaded_session.metadata.name == "test-session"
        assert isinstance(loaded_session.metadata.created_at, datetime)
        assert len(loaded_session.messages) == 1
        assert loaded_session.messages[0]["content"] == "Hello"

def test_session_load_invalid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(session_dir=tmpdir)
        
        # Create a garbage file
        bad_file = Path(tmpdir) / "bad_session.json"
        with open(bad_file, "w") as f:
            f.write("{ invalid json }")
            
        # Try to load it
        session = manager.load_session("bad_session")
        assert session is None

def test_session_load_schema_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(session_dir=tmpdir)
        
        # Create a file with invalid schema (metadata should be a dict)
        bad_file = Path(tmpdir) / "mismatch_session.json"
        with open(bad_file, "w") as f:
            json.dump({"metadata": "not-a-dict"}, f)
            
        # Try to load it
        session = manager.load_session("mismatch_session")
        assert session is None
