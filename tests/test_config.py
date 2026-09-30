from dental_rag.config import Settings


def test_empty_optional_prices_are_accepted(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DENTAL_INPUT_PRICE_PER_MILLION=\nDENTAL_OUTPUT_PRICE_PER_MILLION=\n")
    settings = Settings(_env_file=env_file)
    assert settings.input_price_per_million is None
    assert settings.output_price_per_million is None


def test_provider_secrets_are_excluded_from_report_config(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-gemini-key")
    settings = Settings()
    assert "unit-test" not in settings.model_dump_json()
    assert "API_KEY" not in settings.model_dump_json()
