from src.agentic.mock_backend import MockBackend


def test_semantic_search_finds_api_key_mention():
    hits = MockBackend().semantic_search("API key")
    assert any(hit.text and "API key" in hit.text for hit in hits)


def test_semantic_search_modality_filter():
    hits = MockBackend().semantic_search("dashboard", modality="frame")
    assert hits and all(hit.modality == "frame" for hit in hits)


def test_timestamp_window_inclusive():
    hits = MockBackend().get_chunks_by_timestamp("vid1", 5.0, 10.0)
    ids = {hit.item_id for hit in hits}
    assert "c2" in ids


def test_nearby_expands_around_anchor():
    ids = {hit.item_id for hit in MockBackend().get_nearby_chunks("c3", radius_seconds=3)}
    assert {"c2", "c3", "c4"} <= ids


def test_metadata_unknown_video_returns_empty():
    assert MockBackend().get_video_metadata("nope") == {}
