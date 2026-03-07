from __future__ import annotations

from typing import Optional

import torch


class OATActionTokenizer:
    def __init__(self, checkpoint: str) -> None:
        from oat.tokenizer.oat.tokenizer import OATTok

        self.oat_tokenizer: OATTok = OATTok.from_checkpoint(checkpoint=checkpoint)
        self.oat_tokenizer.eval()

        self.codebook_size = int(self.oat_tokenizer.quantizer.codebook_size)
        self.tokens_per_step = int(self.oat_tokenizer.latent_horizon)

        self._oat_to_hf: Optional[torch.LongTensor] = None
        self._hf_to_oat: dict[int, int] = {}
        self._act_eos_hf_id: Optional[int] = None

    def add_tokens(self, tokenizer, model) -> None:
        token_strings = [f"<act_oat_{i}>" for i in range(self.codebook_size)]
        act_eos = "<act_eos>"

        vocab = tokenizer.get_vocab()
        all_tokens = token_strings + [act_eos]
        new_tokens = [t for t in all_tokens if t not in vocab]
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            model.resize_token_embeddings(len(tokenizer))

        hf_ids = [tokenizer.convert_tokens_to_ids(t) for t in token_strings]
        self._oat_to_hf = torch.tensor(hf_ids, dtype=torch.long)
        self._hf_to_oat = {hf_id: oat_id for oat_id, hf_id in enumerate(hf_ids)}
        self._act_eos_hf_id = int(tokenizer.convert_tokens_to_ids(act_eos))

    def _map_oat_to_hf(self, oat_ids: torch.LongTensor) -> torch.LongTensor:
        if self._oat_to_hf is None:
            raise ValueError("Action tokens are not registered. Call add_tokens() first.")
        return self._oat_to_hf.to(oat_ids.device)[oat_ids]

    def _map_hf_to_oat(self, hf_ids: torch.LongTensor) -> torch.LongTensor:
        flat_hf = hf_ids.detach().to("cpu").reshape(-1).tolist()
        try:
            flat_oat = [self._hf_to_oat[x] for x in flat_hf]
        except KeyError as exc:
            raise ValueError(f"Unknown action token id {int(exc.args[0])}.") from exc
        return torch.tensor(flat_oat, dtype=torch.long, device=hf_ids.device).view_as(hf_ids)

    def tokenize(self, actions: torch.Tensor) -> torch.LongTensor:
        # Delegate real tokenization logic to OAT.
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        if actions.dim() != 3:
            raise ValueError(f"Expected actions shape [B, T, D] or [B, D], got {tuple(actions.shape)}")

        with torch.inference_mode():
            oat_ids = self.oat_tokenizer.tokenize(actions.to(torch.float32))
        return self._map_oat_to_hf(oat_ids)

    def detokenize(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # Delegate real detokenization logic to OAT.
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.dim() != 2:
            raise ValueError(f"Expected token ids shape [B, K] or [K], got {tuple(token_ids.shape)}")

        oat_ids = self._map_hf_to_oat(token_ids)
        with torch.inference_mode():
            return self.oat_tokenizer.detokenize(oat_ids)

    def allowed_hf_token_ids(
        self,
        *,
        device: Optional[torch.device] = None,
        include_eos: bool = True,
    ) -> torch.LongTensor:
        if self._oat_to_hf is None:
            raise ValueError("Action tokens are not registered. Call add_tokens() first.")
        out = self._oat_to_hf
        if include_eos:
            if self._act_eos_hf_id is None:
                raise ValueError("act_eos token is not initialized. Call add_tokens() first.")
            eos = torch.tensor([self._act_eos_hf_id], dtype=torch.long)
            out = torch.cat([out, eos], dim=0)
        if device is not None:
            out = out.to(device)
        return out

    @property
    def act_eos_hf_id(self) -> int:
        if self._act_eos_hf_id is None:
            raise ValueError("act_eos token is not initialized. Call add_tokens() first.")
        return self._act_eos_hf_id
