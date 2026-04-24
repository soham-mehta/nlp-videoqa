from __future__ import annotations

import argparse
import json

from openai import OpenAI

from src.retrieval.modal_client import ModalRetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Check deployed Modal retrieval and vLLM services.")
    parser.add_argument("--generation-base-url", type=str, required=True)
    parser.add_argument("--generation-api-key", type=str, default="EMPTY")
    parser.add_argument("--generation-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--modal-retrieval-app-name", type=str, default="nlp-videoqa-retrieval")
    parser.add_argument("--modal-retrieval-class-name", type=str, default="RetrievalIndex")
    parser.add_argument("--embedding-model", type=str, default="google/siglip2-base-patch16-224")
    parser.add_argument("--modal-index-subdir", type=str, default="indexes/default")
    args = parser.parse_args()

    retrieval = ModalRetrievalService(
        app_name=args.modal_retrieval_app_name,
        class_name=args.modal_retrieval_class_name,
        embedding_model_name=args.embedding_model,
        index_subdir=args.modal_index_subdir,
    )
    retrieval_health = retrieval._instance.healthcheck.remote()

    client = OpenAI(
        base_url=args.generation_base_url,
        api_key=args.generation_api_key,
        timeout=300,
    )
    response = client.chat.completions.create(
        model=args.generation_model,
        messages=[{"role": "user", "content": "Reply with only the word ready."}],
        temperature=0,
        max_tokens=8,
    )

    print(
        json.dumps(
            {
                "retrieval_health": retrieval_health,
                "model_response": response.choices[0].message.content,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
