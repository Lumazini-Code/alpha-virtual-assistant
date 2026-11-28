# Alpha AI (AVA) — Testes da Fase 1 de triagem de memória
# Copyright (C) 2026  Felipe Lumazini
"""Testes do instrumentador de triagem: alinhamento, rótulos e idempotência.

Roda com: python -m unittest discover -s tests
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Modules" / "memory"))

import memory_db  # noqa: E402
import memory_triage as tri  # noqa: E402
from memory_models import TriageLogRequest, TriageSample, TriageTurn  # noqa: E402


class AlignmentTests(unittest.TestCase):
    def test_exact_fuzzy_none(self):
        segs = ["O usuario roda o embeddings server na porta 2002", "estou com pressa"]
        self.assertEqual(tri.align_evidence("roda o embeddings server", segs), (0, "exact"))
        self.assertEqual(tri.align_evidence("usuario roda embeddings server porta 2002", segs)[1], "fuzzy")
        self.assertEqual(tri.align_evidence("algo completamente diferente", segs), (None, "none"))
        self.assertEqual(tri.align_evidence("", segs), (None, "none"))

    def test_normalization_ignores_punctuation_and_case(self):
        self.assertEqual(
            tri.align_evidence("Porta 2002!", ["A PORTA É 2002.", "x"])[1] in ("exact", "fuzzy"),
            True,
        )

    def test_split_segments_tags_role_and_deictic(self):
        segs = tri.split_segments("O servidor é 2002. Ontem mudou.", "Ok, anotei.")
        self.assertTrue(any(s["role"] == "user" for s in segs))
        self.assertTrue(any(s["role"] == "assistant" for s in segs))
        self.assertTrue(any(s["has_deictic"] for s in segs))

    def test_suppression_reason_policy_vs_budget(self):
        self.assertEqual(
            tri.suppression_reason({"text": "estou com pressa", "has_deictic": 0, "n_chars": 16}),
            "policy",
        )
        self.assertEqual(
            tri.suppression_reason({"text": "O servidor de TTS roda na porta 3004", "has_deictic": 0, "n_chars": 38}),
            "budget",
        )


class SampleBuilderTests(unittest.TestCase):
    CTX = {
        "turn_id": "t1",
        "session_id": "s1",
        "scope": "",
        "segments": [
            {"index": 0, "text": "O usuario roda na porta 2002", "role": "user",
             "n_chars": 28, "has_deictic": 0, "hash": "h0"},
            {"index": 1, "text": "estou com pressa", "role": "user",
             "n_chars": 16, "has_deictic": 0, "hash": "h1"},
        ],
        "candidates": [{"id": 7, "text": "O usuario roda na porta 2002", "score": 0.71,
                        "used_in_answer": True, "created_at": 1.0}],
    }

    def test_suppressed_segment_is_unlabeled_never_discard(self):
        samples = tri.build_samples({
            **self.CTX,
            "facts": [{
                "text": "O usuario roda na porta 2002",
                "evidence": "roda na porta 2002",
                "type": "fact", "validity": "temporal",
                "proposed_label": "ADD", "applied_label": "UPDATE",
                "applied_reason": "possible_update:7", "target_ids": [7],
                "related_ids": [], "rank": 1,
            }],
        })
        self.assertEqual(len(samples), 2)
        emitted = [s for s in samples if s["teacher_emitted"]]
        suppressed = [s for s in samples if not s["teacher_emitted"]]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["applied_label"], "UPDATE")
        self.assertEqual(emitted[0]["candidate_id"], 7)
        # Bug de produção: string aqui derruba POST /triage/log com 422.
        self.assertIsInstance(emitted[0]["proposed_targets"], list)
        self.assertEqual(emitted[0]["proposed_targets"], [7])
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["applied_label"], "unlabeled")
        self.assertEqual(suppressed[0]["weight"], 0.0)
        self.assertEqual(suppressed[0]["suppression_reason"], "policy")
        self.assertTrue(all(s["evidence_aligned"] in ("exact", "fuzzy", "none") for s in samples))

    def test_render_labels_are_filled(self):
        samples = tri.build_samples({
            **self.CTX,
            "facts": [{
                "text": "O usuario roda o servidor na porta 2002",
                "evidence": "roda na porta 2002",
                "type": "fact", "validity": "temporal",
                "proposed_label": "ADD", "applied_label": "ADD",
                "target_ids": [], "related_ids": [],
            }],
        })
        emitted = [s for s in samples if s["teacher_emitted"]][0]
        self.assertEqual(emitted["frame_label"], "CONFIG")
        self.assertEqual(emitted["fragment_index"], 0)
        self.assertIn("2002", emitted["slot_value"])

    def test_no_duplicate_on_same_segment_and_candidate(self):
        facts = [{
            "text": "O usuario roda na porta 2002", "evidence": "roda na porta 2002",
            "type": "fact", "proposed_label": "ADD", "applied_label": "ADD",
        } for _ in range(3)]
        samples = tri.build_samples({**self.CTX, "facts": facts})
        self.assertEqual(len(samples), 2)  # 1 emitida + 1 suprimida


class GateSyntheticProbeTests(unittest.TestCase):
    def test_gate_fields_from_real_reasons(self):
        self.assertEqual(
            tri.gate_fields({"stored": False, "reason": "duplicate_exact"})["gate_label"], "NOOP"
        )
        self.assertEqual(
            tri.gate_fields({"stored": False, "reason": "duplicate_semantic:0.93:rerank=0.95"})["gate_label"], "NOOP"
        )
        g = tri.gate_fields({
            "stored": False, "reason": "possible_update:0.89:rerank=0.71",
            "candidate_id": 12, "candidate_score": 0.89, "candidate_rerank_score": 0.71,
        })
        self.assertEqual(g["gate_label"], "UPDATE")
        self.assertEqual(g["gate_candidate_id"], 12)
        self.assertEqual(g["gate_score"], 0.71)  # cross-encoder, não bi-encoder
        self.assertEqual(
            tri.gate_fields({"stored": True, "reason": "ok"})["gate_label"], "ADD"
        )

    def test_synthetic_perturbations_and_expected_labels(self):
        items = tri.synthetic_perturbations(
            "O usuario roda na porta 2002", "roda na porta 2002"
        )
        self.assertEqual(len(items), 4)
        self.assertEqual(
            {i["expected"] for i in items},
            {"UPDATE", "CONFLICT", "NOOP", "DISCARD"},
        )
        self.assertTrue(all("200" in i["text"] for i in items[:3]))
        by_kind = {i["kind"]: i["text"] for i in items}
        self.assertIn("2003", by_kind["version"])
        self.assertIn("2002", by_kind["conflict"])
        self.assertEqual(tri.synthetic_perturbations("sem numeros"), [])

    def test_probe_worthy_only_budget_user_segments(self):
        self.assertTrue(tri.is_probe_worthy(
            {"text": "O servidor roda na porta 2002", "role": "user",
             "n_chars": 32, "has_deictic": 0}
        ))
        self.assertFalse(tri.is_probe_worthy(
            {"text": "estou com pressa", "role": "user", "n_chars": 16, "has_deictic": 0}
        ))
        self.assertFalse(tri.is_probe_worthy(
            {"text": "O servidor roda na porta 2002", "role": "assistant",
             "n_chars": 32, "has_deictic": 0}
        ))

    def test_probe_samples_are_labelled_and_weighted(self):
        seg = {"index": 3, "text": "O servidor roda na porta 2002", "role": "user",
               "n_chars": 32, "has_deictic": 0, "hash": "hp"}
        rows = tri.probe_samples({
            "turn_id": "t1", "session_id": "s1", "scope": "scope1",
            "segments": [seg], "candidates": [], "teacher_variant": "segment_probe",
            "origin": "probe", "created_at": time.time(),
        }, [
            {"segment_index": 3, "text": "O servidor roda na porta 2002", "keep": True},
            {"segment_index": 3, "text": "", "keep": False, "reason": "momentary"},
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["applied_label"], "KEEP")
        self.assertEqual(rows[1]["applied_label"], "DISCARD")
        self.assertTrue(all(r["teacher_variant"] == "segment_probe" for r in rows))
        self.assertTrue(all(r["teacher_emitted"] == 1 for r in rows))

    def test_pair_rows_one_per_top_candidate(self):
        candidates = [
            {"id": 1, "text": "a", "score": 0.9},
            {"id": 2, "text": "b", "score": 0.8},
            {"id": 3, "text": "c", "score": 0.7},
        ]
        samples = tri.build_samples({
            **SampleBuilderTests.CTX, "candidates": candidates,
            "facts": [{
                "text": "O usuario roda na porta 2002", "evidence": "roda na porta 2002",
                "type": "fact", "proposed_label": "ADD", "applied_label": "ADD",
                "target_ids": [1], "related_ids": [],
            }],
        })
        emitted = [s for s in samples if s["teacher_emitted"] and s["segment_index"] == 0]
        self.assertEqual({s["candidate_id"] for s in emitted}, {1, 2, 3})


class ScopeAggregationTests(unittest.TestCase):
    def test_stats_without_scope_sums_all_open_scopes(self):
        import sys
        from types import SimpleNamespace

        dbs = {}
        for name in ("", "proj"):
            path = str(Path(tempfile.mkdtemp()) / f"{name or 'general'}.db")
            db = memory_db.MemoryDB(path)
            dbs[name] = db
        original_db_for = tri._db_for
        tri._db_for = lambda scope=None: dbs.get((scope or "").strip(), dbs[""])
        try:
            for name in ("", "proj"):
                asyncio.run(tri.triage_log(TriageLogRequest(
                    turn=TriageTurn(turn_id=f"t-{name}", session_id="s1",
                                    scope=name, created_at=time.time(),
                                    extractor_model="x"),
                    samples=[TriageSample(
                        turn_id=f"t-{name}", session_id="s1", scope=name,
                        segment_index=0, segment_text="x", segment_hash=f"h-{name}",
                        created_at=time.time(), teacher_emitted=1,
                        applied_label="ADD", proposed_label="ADD",
                    )],
                )))
            sys.modules["memory_state"] = SimpleNamespace(state=SimpleNamespace(
                scopes={"proj": SimpleNamespace()},
                active_scope="proj",
            ))
            stats = asyncio.run(tri.triage_stats())
            self.assertEqual(stats.turns_total, 2)
            self.assertEqual(stats.samples_total, 2)
            self.assertEqual(stats.labelled_total, 2)
        finally:
            tri._db_for = original_db_for
            sys.modules.pop("memory_state", None)
            for db in dbs.values():
                db.close()


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = memory_db.MemoryDB(str(Path(self.tmp.name) / "triage.db"))
        self._original_db_for = tri._db_for
        tri._db_for = lambda scope=None: self.db
        self.target = tri.MEMORY_TRIAGE_TARGET_SAMPLES
        tri.MEMORY_TRIAGE_TARGET_SAMPLES = 2

    def tearDown(self):
        tri._db_for = self._original_db_for
        tri.MEMORY_TRIAGE_TARGET_SAMPLES = self.target
        self.db.close()
        self.tmp.cleanup()

    def _payload(self, turn_id="t1"):
        return TriageLogRequest(
            turn=TriageTurn(
                turn_id=turn_id, session_id="s1", scope="scope1", created_at=time.time(),
                extractor_model="extractor", budget_variant="normal",
                n_facts=1, n_segments=2, n_suppressed=1, status="ok",
            ),
            samples=[
                TriageSample(
                    turn_id=turn_id, session_id="s1", segment_index=0,
                    segment_text="O usuario roda na porta 2002", segment_hash="h0",
                    role="user", n_chars=28, candidate_id=7,
                    teacher_emitted=1, evidence_aligned="exact",
                    applied_label="ADD", proposed_label="ADD", created_at=time.time(),
                ),
                TriageSample(
                    turn_id=turn_id, session_id="s1", segment_index=1,
                    segment_text="estou com pressa", segment_hash="h1",
                    role="user", n_chars=16, teacher_emitted=0,
                    suppression_reason="policy", applied_label="unlabeled",
                    created_at=time.time(),
                ),
            ],
        )

    def test_log_is_idempotent(self):
        first = asyncio.run(tri.triage_log(self._payload()))
        second = asyncio.run(tri.triage_log(self._payload()))
        self.assertTrue(first.stored)
        self.assertTrue(second.stored)
        stats = asyncio.run(tri.triage_stats(scope="scope1"))
        self.assertEqual(stats.turns_total, 1)
        self.assertEqual(stats.samples_total, 2)
        self.assertEqual(stats.labelled_total, 1)
        self.assertEqual(stats.unlabeled_total, 1)

    def test_labels_never_convert_unlabeled_to_negative(self):
        asyncio.run(tri.triage_log(self._payload()))
        labelled = asyncio.run(tri.triage_labels(scope="scope1", include_unlabeled=False))
        everything = asyncio.run(tri.triage_labels(scope="scope1", include_unlabeled=True))
        self.assertEqual(labelled.total, 1)
        self.assertEqual(everything.total, 2)
        self.assertTrue(all(s["applied_label"] != "unlabeled" for s in labelled.samples))
        self.assertTrue(any(s["applied_label"] == "unlabeled" for s in everything.samples))

    def test_readiness_and_recall_gap(self):
        asyncio.run(tri.triage_log(self._payload("t1")))
        pending = asyncio.run(tri.triage_stats(scope="scope1"))
        self.assertFalse(pending.readiness.ready)
        self.assertEqual(pending.readiness.missing, 1)
        # 2º turno com rótulo distinto atinge o alvo de 2 rótulos.
        second = self._payload("t2")
        second.samples[0].applied_label = "UPDATE"
        asyncio.run(tri.triage_log(second))
        ready = asyncio.run(tri.triage_stats(scope="scope1"))
        self.assertTrue(ready.readiness.ready)
        self.assertEqual(ready.readiness.progress, 1.0)

    def test_delete_scope(self):
        asyncio.run(tri.triage_log(self._payload()))
        deleted = asyncio.run(tri.triage_delete(scope="scope1"))
        self.assertGreater(deleted["deleted"], 0)
        stats = asyncio.run(tri.triage_stats(scope="scope1"))
        self.assertEqual(stats.samples_total, 0)

    def test_readiness_counts_weighted_labels_not_raw_rows(self):
        old_target = tri.MEMORY_TRIAGE_TARGET_SAMPLES
        tri.MEMORY_TRIAGE_TARGET_SAMPLES = 10
        try:
            p = self._payload("tw")
            p.samples[0].applied_label = "ADD"
            p.samples[0].origin = "live"
            p.samples[0].weight = 1.0
            p.samples[1].applied_label = "KEEP"
            p.samples[1].origin = "replay"
            p.samples[1].weight = 0.4
            asyncio.run(tri.triage_log(p))
            stats = asyncio.run(tri.triage_stats(scope="scope1"))
            self.assertEqual(stats.labelled_total, 2)
            self.assertAlmostEqual(stats.weighted_labelled_total, 1.4, places=2)
            self.assertEqual(stats.readiness.labelled, 1)
            self.assertEqual(stats.readiness.labelled_raw, 2)
            self.assertAlmostEqual(stats.readiness.progress, 0.14, places=2)
            self.assertEqual(stats.labelled_by_origin["live"], 1)
            self.assertEqual(stats.labelled_by_origin["replay"], 1)
        finally:
            tri.MEMORY_TRIAGE_TARGET_SAMPLES = old_target

    def test_gold_set_is_immutable_and_stratified(self):
        asyncio.run(tri.triage_log(self._payload()))
        candidates = asyncio.run(tri.triage_gold_candidates(limit=10))
        self.assertGreater(candidates.total, 0)
        sid = candidates.samples[0]["sample_id"]
        first = asyncio.run(tri.triage_gold_write(
            __import__("memory_models").TriageGoldRequest(
                sample_id=sid, reviewed_label="ADD", reviewed_by="ana"
            )
        ))
        second = asyncio.run(tri.triage_gold_write(
            __import__("memory_models").TriageGoldRequest(
                sample_id=sid, reviewed_label="DISCARD", reviewed_by="bob"
            )
        ))
        self.assertTrue(first.stored)
        self.assertFalse(second.stored)
        listed = asyncio.run(tri.triage_gold_list())
        self.assertEqual(listed.total, 1)
        self.assertEqual(listed.entries[0]["reviewed_label"], "ADD")
        stats = asyncio.run(tri.triage_stats(scope="scope1"))
        self.assertEqual(stats.gold_total, 1)
        self.assertGreater(stats.gold_target, 0)


if __name__ == "__main__":
    unittest.main()
