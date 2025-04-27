import torch
import pytorch_lightning as pl
import logging
from torch import nn, optim, Tensor
from dataclasses import asdict
from typing import Callable, List, Tuple
from multi_task.metric_calculators import (
    MetricCalculator,
)
from multi_task.metrics_containers import (
    MetricContainer,
)
from multi_task.constants import (
    SKU_EMBEDDING_DIM,
    CATEGORY_EMBEDDING_DIM,
    EVENT_TYPE_EMBEDDING_DIM,
    NAME_EMBEDDING_DIM,
    LSTM_HIDDEN_SIZE,
)
from multi_task.utils import (
    record_embeddings,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class EmbeddingLayer(nn.Module):
    """
    Embedding layer for sku, category, and event type.
    This layer creates separate embeddings for each input feature and concatenates them.

    Args:
        sku_vocab_size (int): Vocabulary size for sku embeddings.
        category_vocab_size (int): Vocabulary size for category embeddings.
        event_type_vocab_size (int): Vocabulary size for event type embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    """

    def __init__(
        self,
        sku_vocab_size: int,
        category_vocab_size: int,
        event_type_vocab_size: int,
    ):
        super().__init__()
        self.sku_embedding = nn.Embedding(
            num_embeddings=sku_vocab_size + 1, 
            embedding_dim=SKU_EMBEDDING_DIM, 
            padding_idx=0
        )
        self.category_embedding = nn.Embedding(
            num_embeddings=category_vocab_size + 1, 
            embedding_dim=CATEGORY_EMBEDDING_DIM, 
            padding_idx=0
        )
        self.event_type_embedding = nn.Embedding(
            num_embeddings=event_type_vocab_size + 1, 
            embedding_dim=EVENT_TYPE_EMBEDDING_DIM, 
            padding_idx=0
        )

    def forward(self, sku: Tensor, category: Tensor, event_type: Tensor) -> Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            sku (Tensor): Input tensor for SKU indices.
            category (Tensor): Input tensor for category indices.
            event_type (Tensor): Input tensor for event type indices.

        Returns:
            Tensor: Concatenated embeddings for SKU, category, and event type.
        """
        sku_emb = self.sku_embedding(sku)
        category_emb = self.category_embedding(category)
        event_type_emb = self.event_type_embedding(event_type)
        return torch.cat([sku_emb, category_emb, event_type_emb], dim=-1)


class SequenceModeling(nn.Module):
    """
    Sequence modeling using LSTM. This class defines an LSTM-based sequence model
    that utilizes the EmbeddingLayer to process input sequences.
    """

    def __init__(
        self, 
        sku_vocab_size: int, 
        category_vocab_size: int, 
        event_type_vocab_size: int, 
    ):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(
            sku_vocab_size=sku_vocab_size,
            category_vocab_size=category_vocab_size,
            event_type_vocab_size=event_type_vocab_size,
        )
        input_size = (
            SKU_EMBEDDING_DIM 
            + CATEGORY_EMBEDDING_DIM 
            + EVENT_TYPE_EMBEDDING_DIM 
            + NAME_EMBEDDING_DIM
            + 1
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            batch_first=True,
        )

    def forward(self, x) -> Tensor:
        """
        Forward pass for sequence modeling.

        Args:
            x: A tuple containing:
                - sequence_sku (Tensor): SKU indices for the sequence.
                - sequence_category (Tensor): Category indices for the sequence.
                - sequence_event_type (Tensor): Event type indices for the sequence.
                - sequence_length (Tensor): Real lengths of the sequences.

        Returns:
            Tensor: Output of the LSTM after processing the input sequences.
        """
        (
            client_id,
            sequence_sku, 
            sequence_category, 
            sequence_price, 
            sequence_name, 
            sequence_event_type, 
            sequence_length 
        ) = x
        # Generate embeddings for the input sequences
        embeddings = self.embedding_layer(sequence_sku, sequence_category, sequence_event_type)
        price_embedding = sequence_price.unsqueeze(-1)
        name_embedding = sequence_name
        embeddings = torch.cat([embeddings, price_embedding, name_embedding], dim=-1)

        # Pass through LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, sequence_length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Average pooling hidden state
        mask = (torch.arange(lstm_output.size(1), device=sequence_length.device)
            < sequence_length.unsqueeze(1))
        lstm_output = (lstm_output * mask.unsqueeze(-1)).sum(dim=1) / sequence_length.unsqueeze(-1)
        
        # Get the last hidden state
        # lstm_output = lstm_output[torch.arange(lstm_output.size(0)), sequence_length - 1]
        
        # Record user representation
        if not self.training:
            client_id = client_id.cpu().numpy()
            embedding = lstm_output.detach().cpu().numpy()
            record_embeddings(client_id, embedding)
        
        return lstm_output


class BottleneckBlock(nn.Module):
    """
    Inverted Bottleneck.
    Taken from "Scaling MLPs: A Tale of Inductive Bias" https://arxiv.org/pdf/2306.13575.pdf.
    The idea is to first expand the input to a wider hidden size, then apply a nonlinearity,
    and finally project back to the original dimension.
    """

    def __init__(self, thin_dim: int, wide_dim: int):
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_size_wide: int,
        hidden_size_thin: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, hidden_size_thin)
        self.ln_input = nn.LayerNorm(normalized_shape=hidden_size_thin)

        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=hidden_size_thin) for _ in range(3)]
        )
        self.bottlenecks = nn.ModuleList(
            [
                BottleneckBlock(thin_dim=hidden_size_thin, wide_dim=hidden_size_wide)
                for _ in range(3)
            ]
        )

        self.ln_output = nn.LayerNorm(normalized_shape=hidden_size_thin)
        self.linear_output = nn.Linear(hidden_size_thin, out_features=output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.ln_input(x)
        for layernorm, bottleneck in zip(self.layernorms, self.bottlenecks):
            x = x + bottleneck(layernorm(x))
        x = self.ln_output(x)
        x = self.linear_output(x)
        return x


class UniversalModel(pl.LightningModule):
    def __init__(
        self,
        sku_vocab_size: int,
        category_vocab_size: int,
        event_type_vocab_size: int,
        output_dims: List[int],
        hidden_size_thin: int,
        hidden_size_wide: int,
        learning_rate: float,
        # metric_calculator: MetricCalculator,
        loss_fn: List[Callable[[Tensor, Tensor], Tensor]],
        # metrics_tracker: List[MetricContainer],
    ) -> None:
        super().__init__()

        torch.manual_seed(1278)
        self.learning_rate = learning_rate
        self.task_nets = nn.ModuleList(
            [
                Net(
                    embedding_dim=LSTM_HIDDEN_SIZE,
                    output_dim=output_dim,
                    hidden_size_thin=hidden_size_thin,
                    hidden_size_wide=hidden_size_wide,
                )
                for output_dim in output_dims
            ]
        )
        self.sequence_modeling = SequenceModeling(
            sku_vocab_size=sku_vocab_size,
            category_vocab_size=category_vocab_size,
            event_type_vocab_size=event_type_vocab_size,
        )
        # self.metric_calculator = metric_calculator
        self.loss_fn = lambda preds, targets: sum(
            loss_fn(pred, target) 
            for pred, target, loss_fn in zip(preds, targets, loss_fn)
        )
        # self.metrics_tracker = metrics_tracker

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.sequence_modeling(x)
        preds = (
            self.task_net(x) for self.task_net in self.task_nets
        )
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage):
        pass
        # self.metric_calculator.to(self.device)

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        # self.metric_calculator.update(
        #     predictions=preds,
        #     targets=y.long(),
        # )

    def on_validation_epoch_end(self) -> None:
        pass
        # metric_container = self.metric_calculator.compute()

        # for metric_name, metric_val in asdict(metric_container).items():
        #     self.log(
        #         metric_name,
        #         metric_val,
        #         prog_bar=True,
        #         logger=True,
        #     )

        # self.metrics_tracker.append(metric_container)
