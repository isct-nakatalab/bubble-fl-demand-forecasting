import copy
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader


class FederatedClient(TypedDict):
    train_loader: DataLoader
    agg_weight: float


class FederatedServer:
    def __init__(self, args, clients: dict[int, FederatedClient], trainer) -> None:
        self.clients = clients
        self.trainer = trainer
        self.fed_epoch = int(args.fed_epoch)
        self.lr = float(args.lr)

    def run_fedavg(
        self, member_ids: list[int], fed_rounds, initial_model: nn.Module | None = None
    ) -> nn.Module | tuple[nn.Module, float] | None:
        member_ids = list(member_ids)
        if not member_ids:
            return initial_model

        rounds = int(fed_rounds)
        global_model = copy.deepcopy(initial_model) if initial_model is not None else self.trainer.create_model()
        global_state = global_model.state_dict()
        final_round_loss = 0.0

        total_weight = sum(self.clients[cid]["agg_weight"] for cid in member_ids) + 1e-9
        weights = {cid: self.clients[cid]["agg_weight"] / total_weight for cid in member_ids}

        for round_idx in range(rounds):
            agg_state = {name: torch.zeros_like(value) for name, value in global_state.items()}
            weighted_round_loss = 0.0
            for cid in member_ids:
                local_model, epoch_losses = self.trainer.train(
                    self.clients[cid]["train_loader"],
                    epochs=self.fed_epoch,
                    lr=self.lr,
                    initial_state=global_state,
                    return_epoch_losses=True,
                )
                if epoch_losses:
                    weighted_round_loss += weights[cid] * epoch_losses[-1]
                for name, value in local_model.state_dict().items():
                    agg_state[name] += value * weights[cid]
            final_round_loss = weighted_round_loss
            global_state = agg_state

        global_model.load_state_dict(global_state)
        return global_model, final_round_loss
