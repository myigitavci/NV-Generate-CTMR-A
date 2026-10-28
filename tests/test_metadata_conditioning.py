"""CPU checks: python -m unittest discover -s tests -v"""
import json
import logging
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi

from scripts.metadata_conditioning import (
    MetadataEncoder, SequenceLabelEncoder, parse_metadata, NUMERICAL, sequence_labels_from_store,
)
from scripts.mrclip_conditioned_unet import MRCLIPConditionedUNet
from scripts.diff_model_train_mrclip import (
    EmbeddingStore, load_filenames_and_indices, load_wrapped_unet, save_checkpoint_mrclip,
    diff_model_train_mrclip,
)


TEXT = (
    "A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, Signa, 1.5), "
    "Acquisition (Description, Sequence, Variant): (Ax T2 FSE (ADNI), SE, SK), "
    "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.08, 4, NONE, 90)"
)


def tiny_unet():
    return DiffusionModelUNetMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, num_channels=(8, 8),
        attention_levels=(False, False), num_res_blocks=1, norm_num_groups=8,
        num_head_channels=8, num_class_embeds=3, include_spacing_input=False,
    )


class MetadataTests(unittest.TestCase):
    def test_legacy_pkl_filename_fallback(self):
        paths = ["/data/sub-1_run-1_T1w_stripped.nii", "/data/sub-2_T2w_stripped_emb.nii.gz",
                 "/data/sub-3_FLAIR.nii.gz", "/data/sub-4_PDw_stripped.nii.gz",
                 "/data/sub-5_T2starw.nii"]
        data = {"embeddings": np.zeros((5, 512)), "sequences": [], "filepaths": paths}
        labels, source = sequence_labels_from_store(data)
        self.assertEqual(labels, ["T1W", "T2W", "FLAIR", "PDW", "T2STARW"])
        self.assertEqual(source, "filepaths_suffix")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            with path.open("wb") as f:
                pickle.dump(data, f)
            store = EmbeddingStore(str(path), "sequence", list(range(5)))
            self.assertEqual(store.sequence_label_source, "filepaths_suffix")
            self.assertEqual(store.embeddings.shape, (5, 5))
        data["sequences"] = ["T2W"]  # Nonempty partial metadata is unsafe.
        with self.assertRaisesRegex(ValueError, "misaligned"):
            sequence_labels_from_store(data)
        data.pop("sequences")
        data["filepaths"][0] = "/data/T1w/sub-1_unknown.nii.gz"
        with self.assertRaisesRegex(ValueError, "explicit coarse contrast suffix"):
            sequence_labels_from_store(data)
        data.pop("filepaths")
        with self.assertRaisesRegex(ValueError, "available keys"):
            sequence_labels_from_store(data)

    def test_sequence_only_store_excludes_all_detailed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pkl"
            data = {"sequences": ["T2W", "T1W", "T2W"],
                    "embeddings": np.full((3, 512), np.nan),
                    "texts": ["invalid text", None, "different scanner and timing"],
                    "labels": [13, 470, 999]}
            with path.open("wb") as f:
                pickle.dump(data, f)
            store = EmbeddingStore(str(path), "sequence", [0, 1])
            np.testing.assert_array_equal(store.get(0), store.get(2))
            self.assertFalse(torch.equal(store.get(0), store.get(1)))
            self.assertEqual(store.sequence_schema["classes"], ["T1W", "T2W"])
            self.assertEqual(tuple(store.embeddings.shape), (3, 2))
            self.assertIsNone(store.metadata_schema)
            encoder = SequenceLabelEncoder(json.loads(json.dumps(store.sequence_schema)))
            np.testing.assert_array_equal(encoder.transform(["t2-w"]), store.embeddings[:1])
            with self.assertRaises(ValueError):
                encoder.transform(["FLAIR"])
            with self.assertRaises(ValueError):
                SequenceLabelEncoder.fit(["SE"])
            with self.assertRaises(ValueError):
                SequenceLabelEncoder.fit([13])
            restored = EmbeddingStore(str(path), "sequence", sequence_schema=store.sequence_schema)
            np.testing.assert_array_equal(restored.embeddings, store.embeddings)
            data.pop("sequences")
            with path.open("wb") as f:
                pickle.dump(data, f)
            with self.assertRaises(ValueError):
                EmbeddingStore(str(path), "sequence", [0, 1])

    def test_parser_retains_nested_parentheses_and_commas(self):
        row = parse_metadata(TEXT.replace("FSE (ADNI)", "FSE, (ADNI)"))
        self.assertEqual(row["description"], "ax t2 fse, (adni)")
        self.assertEqual(row["echo_time"], 0.08)
        self.assertIsNone(row["inversion_time"])
        with self.assertRaises(ValueError):
            parse_metadata("incomplete free text")

    def test_train_only_statistics_unknowns_and_serialization(self):
        second = TEXT.replace("0.08, 4", "0.12, 6")
        encoder = MetadataEncoder.fit([TEXT, second])
        self.assertAlmostEqual(encoder.schema["statistics"]["echo_time"]["mean"], 0.1)
        heldout = TEXT.replace("Signa", "NewScanner").replace("0.08, 4", "0.16, 8")
        transformed = encoder.transform([TEXT, second, heldout])
        self.assertNotIn("newscanner", encoder.schema["categories"]["model"])
        numeric_offset = encoder.input_dim - 2 * len(NUMERICAL)
        np.testing.assert_allclose(transformed[:, numeric_offset + 2], [-1, 1, 3])
        self.assertEqual(transformed[0, numeric_offset + 7], 1)  # Missing TI indicator
        restored = MetadataEncoder(json.loads(json.dumps(encoder.schema)))
        np.testing.assert_array_equal(restored.transform([heldout]), transformed[2:])
        changed = transformed[0] != transformed[2]
        self.assertGreater(changed.sum(), 0)

    def test_store_alignment_and_independence_from_clip_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pkl"
            # Deliberately invalid MR-CLIP values: MLP must never use them.
            data = {"embeddings": np.full((2, 512), np.nan), "texts": [TEXT, TEXT.replace("Signa", "Unseen")]}
            with path.open("wb") as f:
                pickle.dump(data, f)
            store = EmbeddingStore(str(path), "mlp", [0])
            self.assertTrue(torch.isfinite(store.get(1)).all())
            self.assertNotIn("unseen", store.metadata_schema["categories"]["model"])
            with self.assertRaises(ValueError):
                EmbeddingStore(str(path), "mlp", [-1])
            with self.assertRaises(ValueError):
                EmbeddingStore(str(path), "mrclip")
            dataset = Path(directory) / "dataset.json"
            entries = [{"image": "example.nii.gz", "idx": 1}]
            dataset.write_text(json.dumps({"training": entries, "validation": [{"idx": 0}]}))
            self.assertEqual(load_filenames_and_indices(str(dataset)), entries)
            dataset.write_text(json.dumps(entries))
            self.assertEqual(load_filenames_and_indices(str(dataset)), entries)

    def test_projection_gradient_freeze_cfg_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        encoder = MetadataEncoder.fit([TEXT])
        for dim, kind in [(encoder.input_dim, "mlp"), (512, "mrclip"), (5, "sequence")]:
            with self.subTest(conditioning=kind), tempfile.TemporaryDirectory() as directory:
                model = MRCLIPConditionedUNet(tiny_unet(), clip_embed_dim=dim, dropout_p=0)
                # MAISI output conv is zero-initialized. Make it nonzero so a
                # frozen tiny backbone can propagate gradients to its adapter.
                with torch.no_grad():
                    for parameter in model.unet.parameters():
                        if torch.count_nonzero(parameter) == 0 and parameter.ndim > 1:
                            parameter.normal_(0, 0.02)
                model.freeze_unet()
                features = (torch.from_numpy(encoder.transform([TEXT])) if kind == "mlp"
                            else torch.randn(1, dim))
                x, t = torch.randn(1, 1, 8, 8, 8), torch.tensor([1.0])
                output = model(x, t, features)
                self.assertEqual(output.shape, x.shape)
                output.square().mean().backward()
                gradients = [p.grad for p in model.clip_projection.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                self.assertTrue(all(p.grad is None for p in model.unet.parameters()))
                model.zero_grad(set_to_none=True)
                model.dropout_p = 1.0
                model(x, t, features).square().mean().backward()
                self.assertGreater(model.null_embedding.grad.abs().sum(), 0)
                self.assertEqual(model.get_null_embedding(2, torch.device("cpu")).shape, (2, dim))
                info = {"conditioning": kind, "conditioning_dim": dim,
                        "metadata_schema": encoder.schema if kind == "mlp" else None}
                save_checkpoint_mrclip(0, model, 0.1, torch.tensor(2.), directory, "model.pt", 1000, info)
                path = str(Path(directory) / "model.pt")
                saved = torch.load(path, weights_only=False)
                self.assertEqual(saved["conditioning"], kind)
                with patch("scripts.diff_model_train_mrclip.define_instance", side_effect=lambda *_: tiny_unet()):
                    restored = load_wrapped_unet(SimpleNamespace(), torch.device("cpu"),
                                                logging.getLogger("test"), 2, path,
                                                clip_embed_dim=dim, conditioning=kind)
                    self.assertTrue(all(p.requires_grad for p in restored.unet.parameters()))
                    model.eval()
                    restored.eval()
                    torch.testing.assert_close(restored(x, t, features), model(x, t, features))
                    with self.assertRaises(ValueError):
                        load_wrapped_unet(SimpleNamespace(), torch.device("cpu"), logging.getLogger("test"),
                                          1, path, clip_embed_dim=dim,
                                          conditioning="mrclip" if kind == "mlp" else "mlp")

    def test_shared_trainer_runs_both_phases_with_matched_noise(self):
        from monai.networks.schedulers import RFlowScheduler
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.pt"
            torch.save({"unet_state_dict": tiny_unet().state_dict()}, base_path)
            pkl_path = root / "features.pkl"
            with pkl_path.open("wb") as f:
                pickle.dump({"texts": [TEXT, TEXT], "sequences": ["T1W", "T2W"],
                             "embeddings": np.ones((2, 512), np.float32)}, f)
            data_path = root / "dataset.json"
            data_path.write_text(json.dumps({"training": [{"idx": 0}, {"idx": 1}]}))
            loader_images = torch.linspace(-1, 1, 512).reshape(1, 8, 8, 8)

            def loader(items, store, **kwargs):
                return torch.utils.data.DataLoader([
                    {"image": loader_images, "idx": store.get(item["idx"]), "spacing": torch.ones(3)}
                    for item in items
                ], batch_size=1, shuffle=True, num_workers=0)

            noise_draws = {}
            for kind in ("mlp", "mrclip", "sequence"):
                for phase in (1, 2):
                    noise_draws[kind, phase] = []
                    scheduler = RFlowScheduler(num_train_timesteps=1000)
                    add_noise = scheduler.add_noise

                    def record_noise(*, original_samples, noise, timesteps):
                        noise_draws[kind, phase].append(noise.clone())
                        return add_noise(original_samples=original_samples, noise=noise, timesteps=timesteps)

                    scheduler.add_noise = record_noise
                    config = SimpleNamespace(
                        model_dir=str(root / kind), model_filename="model.pt",
                        diffusion_unet_train={"n_epochs": 1, "lr": 1e-3, "batch_size": 1},
                        noise_scheduler={"num_train_timesteps": 1000},
                    )
                    def instance(args, key):
                        return scheduler if key == "noise_scheduler" else tiny_unet()

                    tag = "" if kind == "mrclip" else f"_{kind}"
                    phase1_path = root / kind / f"model_phase1{tag}.pt"
                    with patch("scripts.diff_model_train_mrclip.initialize_distributed", return_value=(0, 1, torch.device("cpu"))), \
                         patch("scripts.diff_model_train_mrclip.load_config", return_value=config), \
                         patch("scripts.diff_model_train_mrclip.prepare_data_mrclip", side_effect=loader), \
                         patch("scripts.diff_model_train_mrclip.define_instance", side_effect=instance):
                        diff_model_train_mrclip(
                            "unused", "unused", "unused", str(pkl_path), str(data_path),
                            conditioning=kind, phase=phase, n_epochs=1, amp=False, tb_dir="", csv_log="",
                            existing_ckpt=str(base_path if phase == 1 else phase1_path),
                            seed=42, proj_lr=1e-3,
                        )
                    result = torch.load(root / kind / f"model_phase{phase}{tag}.pt", weights_only=False)
                    self.assertEqual(result["conditioning"], kind)
                    self.assertTrue(np.isfinite(result["loss"]))
                    if kind == "sequence":
                        self.assertEqual(result["sequence_schema"]["classes"], ["T1W", "T2W"])
                        self.assertIsNone(result["metadata_schema"])
            for phase in (1, 2):
                for kind in ("mlp", "sequence"):
                    self.assertEqual(len(noise_draws[kind, phase]), 2)
                    for noise, clip_noise in zip(noise_draws[kind, phase], noise_draws["mrclip", phase]):
                        torch.testing.assert_close(noise, clip_noise)

    def test_sequence_inference_uses_label_without_metadata_or_clip(self):
        from scripts.diff_model_infer_mrclip import diff_model_infer_mrclip
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = SequenceLabelEncoder.fit(["T1W", "T2W", "PDW", "FLAIR", "T2STARW"])
            model = MRCLIPConditionedUNet(tiny_unet(), clip_embed_dim=encoder.input_dim)
            save_checkpoint_mrclip(0, model, 0.1, torch.tensor(2.), directory, "model.pt", 1000,
                                   {"conditioning": "sequence", "conditioning_dim": encoder.input_dim,
                                    "sequence_schema": encoder.schema})
            torch.save({}, root / "ae.pt")
            config = SimpleNamespace(trained_autoencoder_path=str(root / "ae.pt"),
                                     diffusion_unet_inference={"dim": [8, 8, 8], "spacing": [1, 1, 1]})

            def instance(args, key):
                return torch.nn.Identity() if key == "autoencoder_def" else tiny_unet()

            def generate(args, device, ae, restored, scale, features, *rest):
                self.assertEqual(restored.null_embedding.numel(), 5)
                torch.testing.assert_close(features, torch.from_numpy(encoder.transform(["T2W"])))
                return np.zeros((8, 8, 8), dtype=np.float32)

            with patch("scripts.diff_model_infer_mrclip.initialize_distributed", return_value=(0, 1, torch.device("cpu"))), \
                 patch("scripts.diff_model_infer_mrclip.load_config", return_value=config), \
                 patch("scripts.diff_model_infer_mrclip.define_instance", side_effect=instance), \
                 patch("scripts.diff_model_infer_mrclip.load_clip_text_encoder") as clip_loader, \
                 patch("scripts.diff_model_infer_mrclip.run_inference_mrclip", side_effect=generate):
                output = diff_model_infer_mrclip(
                    "unused", "unused", "unused", str(root / "model.pt"), sequence_label="T2W",
                    metadata_text="deliberately invalid detailed metadata", random_seed=1234, output_dir=directory,
                )
                self.assertTrue(Path(output).name.startswith("sequence_seed1234_"))
                self.assertTrue(Path(output).exists())
                clip_loader.assert_not_called()
                with self.assertRaisesRegex(ValueError, "requires --sequence-label"):
                    diff_model_infer_mrclip("unused", "unused", "unused", str(root / "model.pt"))

    def test_mlp_inference_restores_schema_without_loading_clip(self):
        from scripts.diff_model_infer_mrclip import diff_model_infer_mrclip
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = MetadataEncoder.fit([TEXT])
            model = MRCLIPConditionedUNet(tiny_unet(), clip_embed_dim=encoder.input_dim)
            save_checkpoint_mrclip(0, model, 0.1, torch.tensor(2.), directory, "model.pt", 1000,
                                   {"conditioning": "mlp", "conditioning_dim": encoder.input_dim,
                                    "metadata_schema": encoder.schema})
            torch.save({}, root / "ae.pt")
            config = SimpleNamespace(trained_autoencoder_path=str(root / "ae.pt"),
                                     diffusion_unet_inference={"dim": [8, 8, 8], "spacing": [1, 1, 1]})

            def instance(args, key):
                return torch.nn.Identity() if key == "autoencoder_def" else tiny_unet()

            def generate(args, device, ae, restored, scale, features, *rest):
                self.assertEqual(restored.null_embedding.numel(), encoder.input_dim)
                torch.testing.assert_close(features, torch.from_numpy(encoder.transform([TEXT])))
                self.assertEqual(float(scale), 2.0)
                self.assertEqual(torch.initial_seed(), 1234)
                return np.zeros((8, 8, 8), dtype=np.float32)

            with patch("scripts.diff_model_infer_mrclip.initialize_distributed", return_value=(0, 1, torch.device("cpu"))), \
                 patch("scripts.diff_model_infer_mrclip.load_config", return_value=config), \
                 patch("scripts.diff_model_infer_mrclip.define_instance", side_effect=instance), \
                 patch("scripts.diff_model_infer_mrclip.load_clip_text_encoder") as clip_loader, \
                 patch("scripts.diff_model_infer_mrclip.run_inference_mrclip", side_effect=generate):
                output = diff_model_infer_mrclip(
                    "unused", "unused", "unused", str(root / "model.pt"),
                    metadata_text=TEXT, random_seed=1234, output_dir=directory,
                )
                clip_loader.assert_not_called()
                self.assertTrue(Path(output).exists())
                self.assertTrue(Path(output).name.startswith("mlp_seed1234_"))


if __name__ == "__main__":
    unittest.main()
