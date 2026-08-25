from app.knowledge import embedding


def test_embedding_model_uses_single_cache_entry() -> None:
    # lru_cache(maxsize=1) 保证同一后端进程内只保留一个模型实例。
    info = embedding.embedding_model.cache_info()
    assert info.maxsize == 1


if __name__ == "__main__":
    test_embedding_model_uses_single_cache_entry()
    print("ok")
