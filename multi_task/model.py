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
    QUERY_EMBEDDING_DIM,
    SKU_EMBEDDING_DIM,
    CATEGORY_EMBEDDING_DIM,
    EVENT_TYPE_EMBEDDING_DIM,
    URL_EMBEDDING_DIM,
    NAME_EMBEDDING_DIM,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_BIDIRECTIONAL,
    NUM_CROSS_LAYERS,
    DEEP_HIDDEN_DIMS,
    EMBEDDING_DIM,
    CONTRASTIVE_TEMP,
    CONTRASTIVE_LAMBDA,
)
from multi_task.utils import (
    record_embeddings,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class URLEmbeddingLayer(nn.Module):
    """
    Embedding layer for URL IDs.
    This layer creates embeddings for URL IDs.

    Args:
        url_vocab_size (int): Vocabulary size for URL embeddings.
        embedding_dim (int): Dimensionality of the embeddings.
    """

    def __init__(
        self, 
        url_vocab_size: int
    ):
        super().__init__()
        self.url_embedding = nn.Embedding(
            num_embeddings=url_vocab_size + 1,
            embedding_dim=URL_EMBEDDING_DIM,
            padding_idx=0
        )

    def forward(self, url_ids: Tensor) -> Tensor:
        """
        Forward pass for the URL embedding layer.

        Args:
            url_ids (Tensor): Input tensor for URL ID indices.

        Returns:
            Tensor: Embeddings for URL IDs.
        """
        return self.url_embedding(url_ids)





class SKUEmbeddingLayer(nn.Module):
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


class DCNV2(nn.Module):
    """
    Deep & Cross Network V2 (DCN-V2).
    This block performs explicit feature crossing with improved parameterization
    and combines it with deep layers.

    Args:
        input_dim (int): Dimensionality of the input features.
        num_cross_layers (int): Number of cross layers to apply.
        deep_hidden_dims (List[int]): List of hidden layer dimensions for the deep network.
    """

    def __init__(
        self, 
        input_dim: int,
    ):
        super().__init__()
        self.num_cross_layers = NUM_CROSS_LAYERS
        self.cross_layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=True) for _ in range(NUM_CROSS_LAYERS)]
        )
        
        deep_layers = []
        for in_dim, out_dim in zip([input_dim] + DEEP_HIDDEN_DIMS[:-1], DEEP_HIDDEN_DIMS):
            deep_layers.append(nn.Linear(in_dim, out_dim))
            deep_layers.append(nn.ReLU())
        self.deep_layers = nn.Sequential(*deep_layers)

        self.cross_norm = nn.LayerNorm(input_dim)
        self.deep_norm = nn.LayerNorm(DEEP_HIDDEN_DIMS[-1])

        self.output_layers = nn.Sequential(
            nn.Linear(input_dim + DEEP_HIDDEN_DIMS[-1], input_dim * 4),
            nn.LayerNorm(input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the DCN-V2 block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor after applying cross layers and deep layers.
        """
        x0 = x
        # Cross layers
        for i in range(self.num_cross_layers):
            x = x0 * self.cross_layers[i](x) + x
        cross_output = x
        cross_output = self.cross_norm(cross_output)

        # Log if cross_output contains values greater than 10000 or less than -10000
        if torch.any(cross_output > 1000) or torch.any(cross_output < -1000):
            logger.info(
                "cross_output out of range: max=%s, min=%s", 
                torch.max(cross_output).item(), 
                torch.min(cross_output).item()
            )

        # Deep layers
        deep_output = self.deep_layers(x0)
        deep_output = self.deep_norm(deep_output)

        # Log if deep_output contains values greater than 10000 or less than -10000
        if torch.any(deep_output > 1000) or torch.any(deep_output < -1000):
            logger.info(
                "deep_output out of range: max=%s, min=%s", 
                torch.max(deep_output).item(), 
                torch.min(deep_output).item()
            )

        # Concatenate cross and deep outputs
        combined_output = torch.cat([cross_output, deep_output], dim=-1)
        output = self.output_layers(combined_output)

        # Final output
        return output


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
        url_vocab_size: int
    ):
        super().__init__()
        self.sku_embedding_layer = SKUEmbeddingLayer(
            sku_vocab_size=sku_vocab_size,
            category_vocab_size=category_vocab_size,
            event_type_vocab_size=event_type_vocab_size,
        )
        self.url_embedding_layer = URLEmbeddingLayer(
            url_vocab_size=url_vocab_size,
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
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            bidirectional=LSTM_BIDIRECTIONAL,
        )
        self.lstm_norm = nn.LayerNorm(LSTM_HIDDEN_SIZE)
        self.dcnv2 = DCNV2(
            input_dim=EMBEDDING_DIM,
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
            sequence_url,
            sequence_query,
            sequence_sku_length,
            sequence_url_length,
            sequence_query_length,
        ) = x
        # Generate embeddings for the input sequences
        embeddings = self.sku_embedding_layer(sequence_sku, sequence_category, sequence_event_type)
        price_embedding = sequence_price.unsqueeze(-1)
        name_embedding = sequence_name
        embeddings = torch.cat([embeddings, price_embedding, name_embedding], dim=-1)

        # Pass through LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, sequence_sku_length.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Average pooling hidden state
        mask = (torch.arange(lstm_output.size(1), device=sequence_sku_length.device)
            < sequence_sku_length.unsqueeze(1))
        lstm_output = (lstm_output * mask.unsqueeze(-1)).sum(dim=1) / sequence_sku_length.unsqueeze(-1)
        lstm_output = self.lstm_norm(lstm_output)
        
        # Log if lstm_output contains values greater than 10000 or less than -10000
        if torch.any(lstm_output > 10) or torch.any(lstm_output < -10):
            logger.info(
                "lstm_output out of range: max=%s, min=%s", 
                torch.max(lstm_output).item(), 
                torch.min(lstm_output).item()
            )
        
        
        # Get the last hidden state
        # lstm_output = lstm_output[torch.arange(lstm_output.size(0)), sequence_length - 1]
        
        # Average pooling for url_embedding (query_embedding) based on sequence_url_length (sequence_query_length)
        url_embedding = self.url_embedding_layer(sequence_url)
        query_embedding = sequence_query
        url_mask = (torch.arange(url_embedding.size(1), device=sequence_url_length.device)
                < sequence_url_length.unsqueeze(1))
        url_embedding = (url_embedding * url_mask.unsqueeze(-1)).sum(dim=1) / sequence_url_length.unsqueeze(-1)
        query_mask = (torch.arange(query_embedding.size(1), device=sequence_query_length.device)
                  < sequence_query_length.unsqueeze(1))
        query_embedding = (query_embedding * query_mask.unsqueeze(-1)).sum(dim=1) / sequence_query_length.unsqueeze(-1)


        # Log if url_embedding contains values greater than 10000 or less than -10000
        if torch.any(url_embedding > 10) or torch.any(url_embedding < -10):
            logger.info(
                "url_embedding out of range: max=%s, min=%s", 
                torch.max(url_embedding).item(), 
                torch.min(url_embedding).item()
            )

        # Log if query_embedding contains values greater than 10000 or less than -10000
        if torch.any(query_embedding > 10) or torch.any(query_embedding < -10):
            logger.info(
                "query_embedding out of range: max=%s, min=%s", 
                torch.max(query_embedding).item(), 
                torch.min(query_embedding).item()
            )

        # Concatenate lstm_output, url_embedding, and query_embedding
        combined_feat = torch.cat([lstm_output, url_embedding, query_embedding], dim=-1)
        # Log if combined_feat contains values greater than 10000 or less than -10000
        # if torch.any(combined_feat > 100) or torch.any(combined_feat < -100):
        #     logger.info(
        #         "combined_feat out of range: max=%s, min=%s", 
        #         torch.max(combined_feat).item(), 
        #         torch.min(combined_feat).item()
        #     )
        user_representation = self.dcnv2(combined_feat)
        
        # Log if user_representation contains values greater than 10000 or less than -10000
        if torch.any(user_representation > 10000) or torch.any(user_representation < -10000):
            logger.info(
                "user_representation out of range: max=%s, min=%s", 
                torch.max(user_representation).item(), 
                torch.min(user_representation).item()
            )
        
        # Record user representation
        if not self.training:
            client_id = client_id.cpu().numpy()
            embedding = user_representation.detach().cpu().numpy()
            record_embeddings(client_id, embedding)
        
        return user_representation


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
        url_vocab_size: int,
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
                    embedding_dim=EMBEDDING_DIM,
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
            url_vocab_size=url_vocab_size,
        )
        # self.metric_calculator = metric_calculator
        self.loss_fn = lambda preds, targets: sum(
            loss_fn(pred, target) 
            for pred, target, loss_fn in zip(preds, targets, loss_fn)
        )
        # self.metrics_tracker = metrics_tracker
        self.contrastive_temp = CONTRASTIVE_TEMP
        self.contrastive_lambda = CONTRASTIVE_LAMBDA
        self.nce_fct = nn.CrossEntropyLoss()

    def compute_contrastive_loss(
        self, 
        query: Tensor, 
        key: Tensor,
        normalize: bool = True
    ) -> Tensor:
        """
        Compute the contrastive loss using the InfoNCE loss function.
        Args:
            query (Tensor): The query tensor.
            key (Tensor): The key tensor.
            normalize (bool): Whether to normalize the query and key tensors.
        Returns:
            Tensor: The contrastive loss.
        """
        # Normalize the query and key tensors with L2 normalization
        if normalize:
            query = nn.functional.normalize(query, p=2, dim=1)
            key = nn.functional.normalize(key, p=2, dim=1)
        # Compute the cosine similarity and divede by the temperature
        z = torch.cat((query, key), dim=0)
        sim = torch.mm(z, z.T) / self.contrastive_temp
        # Construct the labels for the NCE loss
        N = query.size(0)
        labels = torch.arange(2 * N, device=z.device)
        labels = (labels + N) % (2 * N)
        # Mask the diagonal elements to avoid self-similarity
        mask = torch.eye(2 * N, device=z.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))
        # Compute the InfoNCE loss
        loss = self.nce_fct(sim, labels)
        return loss

    def forward(self, x) -> Tensor:
        return self.sequence_modeling(x)

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, x_aug1, x_aug2, y = train_batch
        stacked_x = tuple(
            torch.cat([x_elem, x_aug1_elem, x_aug2_elem], dim=0)
            for x_elem, x_aug1_elem, x_aug2_elem in zip(x, x_aug1, x_aug2)
        )
        stacked_user_rep = self.forward(stacked_x)
        user_rep, user_rep_aug1, user_rep_aug2 = torch.chunk(stacked_user_rep, chunks=3, dim=0)

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            user_rep_aug1, 
            user_rep_aug2,
            normalize=True,
        )
        self.log(
            "contrastive_loss", 
            contrastive_loss, 
            on_step=True, 
            prog_bar=True, 
            logger=True
        )
        
        # Compute task-specific loss
        preds = (
            self.task_net(user_rep) for self.task_net in self.task_nets
        )
        task_loss = self.loss_fn(preds, y)
        self.log(
            "task_loss", 
            task_loss, 
            on_step=True, 
            prog_bar=True, 
            logger=True
        )
        total_loss = task_loss + self.contrastive_lambda * contrastive_loss
        self.log(
            "train_loss", 
            total_loss, 
            on_step=True, 
            prog_bar=True, 
            logger=True
        )

        return total_loss

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage):
        pass
        # self.metric_calculator.to(self.device)

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        user_rep = self.forward(x)
        preds = (
            self.task_net(user_rep) for self.task_net in self.task_nets
        )
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
