"""Validate fail-closed external-service scaffold behavior."""

import unittest
from pathlib import Path

from temporal_alignment.errors import ContractError, EvidenceGateError
from temporal_alignment.ServiceAdapters.contracts import ExternalAssetSpec
from temporal_alignment.ServiceAdapters.huggingface import HuggingFaceAdapter
from temporal_alignment.ServiceAdapters.inference import (
    HuggingFaceInferenceAdapter,
    InferencePlan,
)
from temporal_alignment.ServiceAdapters.kaggle import KaggleAdapter


def _spec(provider: str, resource_id: str, revision: str = "a" * 40) -> ExternalAssetSpec:
    return ExternalAssetSpec(
        provider=provider,
        resource_id=resource_id,
        source_url="https://example.org/asset",
        accessed_on="2026-08-24",
        license_id="MIT-test-fixture",
        terms_url="https://example.org/terms",
        revision=revision,
        sha256="0" * 64,
        reviewer="test-reviewer",
        data_role="reference-only",
    )


def _inference_plan(model_revision: str, privacy_class: str) -> InferencePlan:
    return InferencePlan(
        provider="huggingface",
        provider_route="hf-inference",
        model_id="org/model",
        model_revision=model_revision,
        task="feature-extraction",
        input_schema="text-v1",
        output_schema="embedding-v1",
        privacy_class=privacy_class,
        timeout_seconds=30.0,
        retry_budget=1,
        max_input_units=512,
        normalization="l2",
    )


class ServiceAdapterTests(unittest.TestCase):
    def test_huggingface_rejects_mutable_revision(self) -> None:
        with self.assertRaises(ContractError):
            HuggingFaceAdapter().build_plan(_spec("huggingface", "org/repo", "main"), Path("cache"))

    def test_kaggle_requires_version_and_fetch_stays_disabled(self) -> None:
        adapter = KaggleAdapter()
        with self.assertRaises(ContractError):
            adapter.build_plan(_spec("kaggle", "owner/dataset"), Path("cache"))
        plan = adapter.build_plan(_spec("kaggle", "owner/dataset/versions/1"), Path("cache"))
        with self.assertRaises(EvidenceGateError):
            adapter.fetch(plan)

    def test_remote_inference_requires_public_data_and_immutable_revision(self) -> None:
        with self.assertRaises(ContractError):
            _inference_plan(model_revision="main", privacy_class="public")
        with self.assertRaises(EvidenceGateError):
            _inference_plan(model_revision="a" * 40, privacy_class="personal")

    def test_remote_inference_execution_stays_disabled(self) -> None:
        plan = InferencePlan(
            provider="huggingface",
            provider_route="hf-inference",
            model_id="org/model",
            model_revision="a" * 40,
            task="feature-extraction",
            input_schema="text-v1",
            output_schema="embedding-v1",
            privacy_class="public",
            timeout_seconds=30.0,
            retry_budget=1,
            max_input_units=512,
            normalization="l2",
        )
        with self.assertRaises(EvidenceGateError):
            HuggingFaceInferenceAdapter().execute(plan, "public test payload")


if __name__ == "__main__":
    unittest.main()
