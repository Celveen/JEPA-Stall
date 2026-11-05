"""Train the approved B arms through the original Trainer/optimizers.

Only the IDM arm adds parameters, to the action/proprio AdamW optimizer at its
unchanged LR. Initialization and sampler hashes are recorded before updates.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hydra
import torch
from omegaconf import OmegaConf
from train import Trainer


def module_hash(module):
    module = getattr(module, "module", module)
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EffectTrainer(Trainer):
    def init_models(self):
        self.idm_head = None
        if self.cfg.model.effect_mode == "idm":
            self._keys_to_save.append("idm_head")
        super().init_models()
        if self.model.idm_head is not None:
            head = self.idm_head if self.idm_head is not None else self.model.idm_head
            self.idm_head = self.accelerator.prepare(head)
            self.model.idm_head = self.idm_head
        if self.accelerator.is_main_process:
            record = {"epoch_at_start": self.epoch, "mode": self.cfg.model.effect_mode,
                "modules": {k: module_hash(getattr(self, k)) for k in
                            ("encoder", "predictor", "action_encoder", "proprio_encoder")},
                "samplers": {k: hashlib.sha256(v.slices.tobytes()).hexdigest() for k,v in self.datasets.items()},
                "dataset_lengths": {k: len(v) for k,v in self.datasets.items()},
                "aux_parameters": sum(p.numel() for p in self.idm_head.parameters()) if self.idm_head else 0,
                "effect_lambda": float(self.cfg.model.effect_lambda), "world_size": self.accelerator.num_processes,
                "effective_batch_size": self.cfg.training.batch_size}
            expected = self.cfg.get("effect_expected_init")
            if expected and self.epoch == 0:
                ref = json.loads(Path(expected).read_text())
                if record["modules"] != ref["modules"]:
                    raise RuntimeError("Backbone initialization does not match registered calibration")
            Path("experiment_initialization.json").write_text(json.dumps(record, indent=2)+"\n")
        self.accelerator.wait_for_everyone()

    def run(self):
        steps = int(self.cfg.get("effect_smoke_steps", 0))
        if not steps:
            return super().run()
        self.model.train()
        initial = module_hash(self.idm_head) if self.idm_head is not None else None
        rows = []
        for i, (obs, act, state) in enumerate(self.dataloaders["train"]):
            self.encoder_optimizer.zero_grad()
            self.predictor_optimizer.zero_grad()
            self.action_encoder_optimizer.zero_grad()
            _, _, _, loss, lc = self.model(obs, act)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite smoke loss")
            self.accelerator.backward(loss)
            for parameter in self.model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite smoke gradient")
            self.encoder_optimizer.step()
            self.predictor_optimizer.step()
            self.action_encoder_optimizer.step()
            gathered = self.accelerator.gather_for_metrics({k:v.detach() for k,v in lc.items()})
            rows.append({k:float(v.float().mean()) for k,v in gathered.items()})
            if i+1 >= steps:
                break
        if self.idm_head is not None and module_hash(self.idm_head) == initial:
            raise RuntimeError("IDM head optimizer did not update parameters")
        self.save_ckpt(tag="smoke")
        if self.accelerator.is_main_process:
            Path("smoke.json").write_text(json.dumps({"steps":steps,"world_size":self.accelerator.num_processes,
                "rows":rows,"idm_head_updated":self.idm_head is not None},indent=2)+"\n")
        self.accelerator.wait_for_everyone()

    def init_optimizers(self):
        super().init_optimizers()
        if self.idm_head is not None:
            self.action_encoder_optimizer.add_param_group({"params": list(self.idm_head.parameters()),
                                                           "lr": self.cfg.training.action_encoder_lr})
            old = getattr(self, "_loaded_optim_state", {}).get("action_encoder_optimizer")
            if old is not None:
                # Parent restoration precedes the extra parameter group; restore
                # again now that the original parameter-group structure exists.
                self.action_encoder_optimizer.load_state_dict(old)


@hydra.main(config_path="../conf", config_name="train_effect_b0", version_base=None)
def main(cfg):
    trainer = EffectTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
