import json

from src.config.settings import PathsConfig
from src.data.loaders import discover_video_ids, enumerate_frames_for_video, load_transcript_chunks_for_video


def test_data_loaders_fall_back_to_hf_data(tmp_path):
    repo_root = tmp_path
    data_root = repo_root / "data"
    hf_root = repo_root / "hf_data"
    (data_root / "transcripts").mkdir(parents=True)
    (data_root / "frames").mkdir(parents=True)
    (data_root / "metadata" / "frames").mkdir(parents=True)
    (data_root / "indexes" / "default").mkdir(parents=True)
    (hf_root / "transcripts").mkdir(parents=True)
    (hf_root / "metadata" / "frames").mkdir(parents=True)
    (hf_root / "frames" / "vid1").mkdir(parents=True)

    (hf_root / "frames" / "vid1" / "0000001000.jpg").write_bytes(b"jpg")
    (hf_root / "transcripts" / "vid1.json").write_text(
        json.dumps(
            {
                "video_id": "vid1",
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": "hello world"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (hf_root / "metadata" / "frames" / "vid1.json").write_text(
        json.dumps(
            {
                "video_id": "vid1",
                "frames": [
                    {
                        "timestamp_sec": 1.0,
                        "frame_index": 1,
                        "image_path": "frames/vid1/0000001000.jpg",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    paths = PathsConfig.from_repo_root(repo_root)

    assert discover_video_ids(paths) == ["vid1"]

    chunks = load_transcript_chunks_for_video(paths, "vid1")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"

    frames = enumerate_frames_for_video(paths, "vid1")
    assert len(frames) == 1
    assert frames[0].frame_path == str(hf_root / "frames" / "vid1" / "0000001000.jpg")
