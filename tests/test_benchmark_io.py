import json

from src.benchmark.io import load_benchmark_items


def test_load_benchmark_items_supports_multimodal_v2_json(tmp_path):
    benchmark_path = tmp_path / "multimodal_benchmark_v2.json"
    benchmark_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "vid1",
                    "video_title": "Demo Video",
                    "items": [
                        {
                            "question_id": "q1",
                            "question_type": "cross_modal",
                            "question": "What is shown?",
                            "ideal_answer": "A terminal next to a subtitle.",
                            "gold_evidence": [
                                {
                                    "modality": "text",
                                    "timestamp_start": 10.0,
                                    "timestamp_end": 12.0,
                                    "transcript_chunk_ids": ["seg_001"],
                                    "frame_ids": [],
                                },
                                {
                                    "modality": "image",
                                    "timestamp_start": 11.0,
                                    "timestamp_end": 12.0,
                                    "transcript_chunk_ids": [],
                                    "frame_ids": ["frames/vid1/0000011000.jpg"],
                                },
                            ],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    items = load_benchmark_items(str(benchmark_path))

    assert len(items) == 1
    item = items[0]
    assert item.question_id == "q1"
    assert item.video_id == "vid1"
    assert item.gold_answer == "A terminal next to a subtitle."
    assert item.question_type == "multimodal"
    assert item.question_type_raw == "cross_modal"
    assert [gold.modality for gold in item.gold_evidence] == ["text", "frame"]
    assert item.gold_evidence[0].source_ids == ["seg_001"]
    assert item.gold_evidence[1].source_ids == ["frames/vid1/0000011000.jpg"]
