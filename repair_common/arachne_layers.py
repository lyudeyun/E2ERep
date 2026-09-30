"""Shared target-layer handling for Arachne repair implementations."""

import torch.nn as nn


class ArachneTargetLayerMixin:
    """Resolve target layers and collect their per-weight gradients."""

    def _find_target_layers(self, model):
        if self.target_layer:
            named_modules = dict(model.named_modules())
            missing = [name for name in self.target_layer if name not in named_modules]
            if missing:
                raise ValueError(f"Target layers not found in model: {missing}")
            return [(name, named_modules[name]) for name in self.target_layer]

        linear_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            raise ValueError("No Linear layer found in model")

        target_name, target_layer = linear_layers[-1]
        self.target_layer = [target_name]
        return [(target_name, target_layer)]

    def _collect_target_gradient_candidates(self, model):
        candidates = []
        for layer_name, layer in self._find_target_layers(model):
            if layer.weight.grad is None:
                print(f"Warning: No gradient for {layer_name}, skipping")
                continue

            grad = layer.weight.grad.detach().cpu().numpy()
            for j in range(grad.shape[0]):
                for i in range(grad.shape[1]):
                    candidates.append([layer_name, i, j, float(abs(grad[j, i]))])

        candidates.sort(key=lambda item: item[3], reverse=True)
        return candidates
