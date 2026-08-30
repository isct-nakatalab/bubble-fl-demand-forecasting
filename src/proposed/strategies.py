import copy
import logging
import random
from typing import Any

import numpy as np
from torch import nn

logger = logging.getLogger(__name__)


class StrategyProvider:
    def __init__(self, args, trainer, server, clients: dict, all_local_models: dict[int, nn.Module]):
        self.trainer = trainer
        self.server = server
        self.clients = clients
        self.all_local_models = all_local_models
        self.fed_rounds = int(args.fed_rounds)
        self.adaptive_iter = int(args.adaptive_iter)
        self.regen_iter = int(args.regen_iter)
        self.bubble_eval_dataset = args.bubble_eval_dataset
        self.adaptive_mode = args.adaptive_mode
        self.restructure_mode = args.restructure_mode
        self.min_improvement = float(args.min_improvement)
        self.cached_fedavg_model: nn.Module | None = None
        self.bubble_rows: list[dict[str, Any]] = []

    def _get_fedavg_model(self) -> nn.Module:
        if self.cached_fedavg_model is None:
            self.cached_fedavg_model, _ = self.server.run_fedavg(list(self.clients.keys()), self.fed_rounds)
        return self.cached_fedavg_model

    def _evaluate_target(self, model: nn.Module, target_id: int, phase: str) -> tuple[float, float, float, float]:
        return self.trainer.evaluate(model, self.clients[target_id], phase=phase)

    def _append_bubble_row(
        self,
        target_id: int,
        phase: str,
        trial: int,
        members: list[int],
        loss: float | None,
        rmse: float,
        r2: float,
    ) -> None:
        self.bubble_rows.append(
            {
                "client": target_id,
                "phase": phase,
                "trial": trial,
                "bubble": list(members),
                "loss": None if loss is None else float(loss),
                "rmse": float(rmse),
                "r2": float(r2),
            }
        )

    def _build_stage_result(self, model: nn.Module, target_id: int, members: list[int]) -> dict:
        return {
            "rmse": self._evaluate_target(model, target_id, phase="test")[1],
            "members": list(members),
            "model": model,
        }

    def execute(self, target_id: int) -> dict:
        logger.info(f"[Client {target_id}] FedAvg")
        fedavg_model = self._get_fedavg_model()
        stage_results = {"fedavg": self._build_stage_result(fedavg_model, target_id, list(self.clients.keys()))}
        logger.info(f"[Client {target_id}] Adaptive bubble construction")
        best_members = [target_id]
        best_model = self.all_local_models[target_id]
        _, best_rmse, local_r2, _ = self._evaluate_target(best_model, target_id, phase=self.bubble_eval_dataset)
        self._append_bubble_row(target_id, "bubble_build_init", 0, best_members, None, best_rmse, local_r2)
        best_rmse, best_members, best_model = self._adaptive_phase(target_id, best_rmse, best_members, best_model)
        stage_results["adaptive"] = self._build_stage_result(best_model, target_id, best_members)
        if self.regen_iter > 0:
            logger.info(f"[Client {target_id}] Bubble restructure")
            best_rmse, best_members, best_model = self._restructuring_phase(
                target_id, best_rmse, best_members, best_model
            )
            stage_results["proposal"] = self._build_stage_result(best_model, target_id, best_members)
        return stage_results

    def _rank_adaptive_candidates(
        self, target_id: int, reference_model: nn.Module, excluded_ids: set[int]
    ) -> list[int]:
        candidate_ids = [cid for cid in self.clients if cid not in excluded_ids]
        ranked = []
        for candidate_id in candidate_ids:
            candidate_model = self.all_local_models[candidate_id]
            if self.adaptive_mode == "other_on_target":
                eval_model = candidate_model
                eval_id = target_id
            elif self.adaptive_mode == "target_on_other":
                eval_model = reference_model
                eval_id = candidate_id
            else:
                raise ValueError(f"Unknown adaptive_mode: {self.adaptive_mode}")
            rmse = self.trainer.evaluate(eval_model, self.clients[eval_id], phase=self.bubble_eval_dataset)[1]
            ranked.append((candidate_id, rmse))
        ranked.sort(key=lambda item: item[1])
        return [candidate_id for candidate_id, _ in ranked]

    def _select_restructure_candidate(
        self, target_id: int, reference_model: nn.Module, excluded_ids: set[int]
    ) -> int | None:
        candidate_ids = [cid for cid in self.clients if cid not in excluded_ids]
        if self.restructure_mode == "random":
            return random.choice(candidate_ids) if candidate_ids else None
        if self.restructure_mode != "residual":
            raise ValueError(f"Unknown restructure_mode: {self.restructure_mode}")

        target_loader = self.clients[target_id]["train_loader"]
        preds, trues = self.trainer.predict_loader(reference_model, target_loader, None)
        residuals = (trues - preds).reshape(-1)
        best_candidate = None
        best_corr = -1.0
        for candidate_id in candidate_ids:
            candidate_model = self.all_local_models[candidate_id]
            candidate_preds, _ = self.trainer.predict_loader(candidate_model, target_loader, None)
            candidate_signal = candidate_preds.reshape(-1)
            if len(residuals) < 2 or len(candidate_signal) < 2:
                continue
            residual_std = np.std(residuals)
            signal_std = np.std(candidate_signal)
            if signal_std > residual_std * 3.0:
                continue
            if signal_std <= 1e-6 or residual_std <= 1e-6:
                continue
            corr = np.corrcoef(residuals, candidate_signal)[0, 1]
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_candidate = candidate_id
        return best_candidate

    def _adaptive_phase(
        self, target_id: int, rmse: float, members: list[int], model: nn.Module
    ) -> tuple[float, list[int], nn.Module]:
        best_rmse = rmse
        best_members = list(members)
        best_model = model
        adaptive_candidates = self._rank_adaptive_candidates(target_id, best_model, set(best_members))[
            : self.adaptive_iter
        ]
        for trial_idx, candidate_id in enumerate(adaptive_candidates, start=1):
            trial_members = best_members + [candidate_id]
            trial_model, trial_loss = self.server.run_fedavg(trial_members, self.fed_rounds)
            _, trial_rmse, trial_r2, _ = self._evaluate_target(trial_model, target_id, phase=self.bubble_eval_dataset)
            self._append_bubble_row(
                target_id, "bubble_build", trial_idx, trial_members, trial_loss, trial_rmse, trial_r2
            )
            if trial_rmse + self.min_improvement < best_rmse:
                best_rmse = trial_rmse
                best_members = trial_members
                best_model = trial_model
        return best_rmse, best_members, best_model

    def _restructuring_phase(
        self, target_id: int, rmse: float, members: list[int], model: nn.Module
    ) -> tuple[float, list[int], nn.Module]:
        best_rmse = rmse
        best_members = list(members)
        best_model = model
        final_safe_model = copy.deepcopy(model)

        for trial_idx in range(1, self.regen_iter + 1):
            trial_members = copy.deepcopy(best_members)
            removable = [cid for cid in trial_members if cid != target_id]
            if removable:
                drop_id = random.choice(removable)
                trial_members.remove(drop_id)
                trial_model, trial_loss = self.server.run_fedavg(trial_members, self.fed_rounds)
                _, trial_rmse, trial_r2, _ = self._evaluate_target(
                    trial_model, target_id, phase=self.bubble_eval_dataset
                )
                self._append_bubble_row(
                    target_id, "bubble_restruct_drop", trial_idx, trial_members, trial_loss, trial_rmse, trial_r2
                )
                if trial_rmse + self.min_improvement < best_rmse:
                    best_rmse = trial_rmse
                    best_members = trial_members
                    best_model = trial_model
                    final_safe_model = copy.deepcopy(trial_model)
            best_add = self._select_restructure_candidate(target_id, best_model, excluded_ids=set(best_members))
            if best_add is None:
                continue
            trial_members = best_members + [best_add]
            trial_model, trial_loss = self.server.run_fedavg(trial_members, self.fed_rounds)
            _, trial_rmse, trial_r2, _ = self._evaluate_target(trial_model, target_id, phase=self.bubble_eval_dataset)
            self._append_bubble_row(
                target_id, "bubble_restruct_add", trial_idx, trial_members, trial_loss, trial_rmse, trial_r2
            )
            if trial_rmse + self.min_improvement < best_rmse:
                best_rmse = trial_rmse
                best_members = trial_members
                best_model = trial_model
                final_safe_model = copy.deepcopy(trial_model)
        return best_rmse, best_members, final_safe_model
