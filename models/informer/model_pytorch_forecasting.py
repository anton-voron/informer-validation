from typing import Dict, List, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import functools
import operator
from copy import copy

from pytorch_forecasting.models.nn import MultiEmbedding
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss

from informer.attention import (
    FullAttention, ProbSparseAttention, AttentionLayer)
from informer.embedding import TokenEmbedding, PositionalEmbedding
from informer.encoder import (
    Encoder,
    EncoderLayer,
    SelfAttentionDistil,
)
from informer.decoder import Decoder, DecoderLayer


class Informer(BaseModelWithCovariates):

    def __init__(
            self,
            d_model=256,
            d_fully_connected=512,
            n_attention_heads=2,
            n_encoder_layers=2,
            n_decoder_layers=1,
            dropout=0.1,
            attention_type="prob",
            activation="gelu",
            factor=5,
            mix_attention=False,
            output_attention=False,
            distil=True,
            x_reals: List[str] = [],
            x_categoricals: List[str] = [],
            static_categoricals: List[str] = [],
            static_reals: List[str] = [],
            time_varying_reals_encoder: List[str] = [],
            time_varying_reals_decoder: List[str] = [],
            time_varying_categoricals_encoder: List[str] = [],
            time_varying_categoricals_decoder: List[str] = [],
            embedding_sizes: Dict[str, Tuple[int, int]] = {},
            embedding_paddings: List[str] = [],
            embedding_labels: Dict[str, np.ndarray] = {},
            categorical_groups: Dict[str, List[str]] = {},
            output_size: Union[int, List[int]] = 1,
            loss=None,
            logging_metrics: nn.ModuleList = None,
            **kwargs):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            **kwargs)
        self.save_hyperparameters(ignore=['loss'])
        self.attention_type = attention_type

        assert not static_reals
        assert not static_categoricals

        self.cat_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )

        self.enc_real_embeddings = TokenEmbedding(
            len(time_varying_reals_encoder), d_model)
        self.enc_positional_embeddings = PositionalEmbedding(d_model)

        self.dec_real_embeddings = TokenEmbedding(
            len(time_varying_reals_decoder), d_model)
        self.dec_positional_embeddings = PositionalEmbedding(d_model)

        Attention = ProbSparseAttention \
            if attention_type == "prob" else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attention(False, factor, attention_dropout=dropout,
                                  output_attention=output_attention),
                        d_model,
                        n_attention_heads,
                        mix=False,
                    ),
                    d_model,
                    d_fully_connected,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_encoder_layers)
            ],
            [SelfAttentionDistil(d_model) for _ in range(
                n_encoder_layers - 1)] if distil else None,
            nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attention(True, factor, attention_dropout=dropout,
                                  output_attention=False),
                        d_model,
                        n_attention_heads,
                        mix=mix_attention,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False),
                        d_model,
                        n_attention_heads,
                        mix=False,
                    ),
                    d_model,
                    d_fully_connected,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_decoder_layers)
            ],
            nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, output_size)

    def forward(
            self,
            x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        decoder_length = x['decoder_lengths'].max()

        enc_out =\
            self.enc_real_embeddings(x['encoder_cont']) +\
            self.enc_positional_embeddings(x['encoder_cont']) +\
            functools.reduce(operator.add, [emb for emb in self.cat_embeddings(
                x['encoder_cat']).values()])
        enc_out, attentions = self.encoder(enc_out)

        # Hacky solution to get only known reals,
        # they are always stacked first.
        # TODO: Make sure no unknown reals are passed to decoder.
        dec_out =\
            self.dec_real_embeddings(x['decoder_cont'][..., :len(
                self.hparams.time_varying_reals_decoder)]) +\
            self.dec_positional_embeddings(x['decoder_cont']) +\
            functools.reduce(operator.add, [emb for emb in self.cat_embeddings(
                x['decoder_cat']).values()])
        dec_out = self.decoder(dec_out, enc_out)

        output = self.projection(dec_out)
        output = output[:, -decoder_length:, :]
        output = self.transform_output(
            output, target_scale=x['target_scale'])
        return self.to_network_output(prediction=output)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        **kwargs
    ):
        new_kwargs = copy(kwargs)
        new_kwargs.update(cls.deduce_default_output_parameters(
            dataset, kwargs, QuantileLoss()))

        return super().from_dataset(dataset, **new_kwargs)
