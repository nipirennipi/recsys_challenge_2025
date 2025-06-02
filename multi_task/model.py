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
    SKU_ID_EMBEDDING_DIM,
    SKU_CATEGORY_EMBEDDING_DIM,
    EVENT_TYPE_EMBEDDING_DIM,
    URL_EMBEDDING_DIM,
    NAME_EMBEDDING_DIM,
    # LSTM_HIDDEN_SIZE,
    # LSTM_NUM_LAYERS,
    # LSTM_DROPOUT,
    # LSTM_BIDIRECTIONAL,
    SKU_EMBEDDING_DIM,
    SASREC_NUM_HEADS,
    SASREC_NUM_LAYERS,
    NUM_CROSS_LAYERS,
    DEEP_HIDDEN_DIMS,
    EMBEDDING_DIM,
    CONTRASTIVE_TEMP,
    CONTRASTIVE_LAMBDA,
    MLP_PROJECTION_DIM,
    TIME_FEAT_DIM,
    MAX_SEQUENCE_LENGTH,
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
            embedding_dim=SKU_ID_EMBEDDING_DIM, 
            padding_idx=0
        )
        self.category_embedding = nn.Embedding(
            num_embeddings=category_vocab_size + 1, 
            embedding_dim=SKU_CATEGORY_EMBEDDING_DIM, 
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


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec).
    This model uses self-attention to model user-item interactions in sequences.

    Args:
        hidden_dim (int): Dimensionality of the embeddings.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_seq_len (int): Maximum sequence length.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=num_layers,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_emb: Tensor, seq_len: Tensor) -> Tensor:
        """
        Forward pass for SASRec.

        Args:
            seq_emb (Tensor): Input tensor of item indices (batch_size, max_seq_len, emb_dim).
            seq_len (Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            Tensor: Last hidden state (batch_size, emb_dim).
        """
        batch_size, max_seq_len, _ = seq_emb.size()
        positions = (
            torch.arange(max_seq_len, device=seq_emb.device)
            .unsqueeze(0)
            .expand(batch_size, -1)    
        )
        
        # Generate embeddings
        pos_emb = self.position_embedding(positions)
        x = self.layer_norm(seq_emb + pos_emb)
        x = self.dropout(x)

        # Create attention mask
        attn_mask = (
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, device=seq_emb.device), 
                diagonal=1,
            ).bool()
        ) 
        
        # Create key padding mask
        src_key_padding_mask = positions >= seq_len.unsqueeze(1)
        
        # Pass through transformer
        x = self.transformer(
            x.permute(1, 0, 2), 
            src_key_padding_mask=src_key_padding_mask, 
            mask=attn_mask
        )
        x = x.permute(1, 0, 2)

        # Gather the last relevant hidden state for each sequence
        last_hidden_state = x[torch.arange(batch_size), seq_len - 1]

        return last_hidden_state


class AveragePooling(nn.Module):
    """
    Average Pooling Layer.
    This layer computes the average of embeddings over the sequence length.

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, seq_emb: Tensor, seq_len: Tensor) -> Tensor:
        """
        Forward pass for average pooling.

        Args:
            seq_emb (Tensor): Input tensor of shape (batch_size, max_seq_len, emb_dim).
            seq_len (Tensor): Lengths of each sequence in the batch (batch_size).

        Returns:
            Tensor: Pooled embeddings of shape (batch_size, emb_dim).
        """
        # Mask out padding positions
        mask = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(seq_emb)  # Expand mask to match seq_emb shape
        masked_seq_emb = seq_emb * mask

        # Compute sum of embeddings and divide by sequence length
        pooled_emb = masked_seq_emb.sum(dim=1) / seq_len.unsqueeze(-1)
        return pooled_emb


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
        url_vocab_size: int,
        item_stat_feat_dim: int,
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
        self.sku_seq_encoder = SASRec(
            hidden_dim=SKU_EMBEDDING_DIM,
            num_heads=SASREC_NUM_HEADS,
            num_layers=SASREC_NUM_LAYERS,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            dropout=0.1,
        )
        self.url_seq_encoder = SASRec(
            hidden_dim=URL_EMBEDDING_DIM,
            num_heads=SASREC_NUM_HEADS,
            num_layers=SASREC_NUM_LAYERS,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            dropout=0.1,
        )
        self.query_seq_encoder = AveragePooling()
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
            sequence_stat_feat,
            sequence_event_type, 
            sequence_time_feat,
            sequence_url,
            sequence_query,
            sequence_sku_length,
            sequence_url_length,
            sequence_query_length,
            user_features,
        ) = x
        # Generate embeddings for the input sequences
        sku_embedding = self.sku_embedding_layer(sequence_sku, sequence_category, sequence_event_type)
        price_embedding = sequence_price.unsqueeze(-1)
        name_embedding = sequence_name
        stat_feat_embedding = sequence_stat_feat
        time_feat_embedding = sequence_time_feat
        sku_embedding = torch.cat(
            [sku_embedding, price_embedding, name_embedding, stat_feat_embedding, time_feat_embedding], 
            dim=-1
        )

        # Pass through SASRec, respectively for sku, url, and query
        sku_seq_output = self.sku_seq_encoder(sku_embedding, sequence_sku_length)
        
        url_embedding = self.url_embedding_layer(sequence_url)
        url_seq_output = self.url_seq_encoder(url_embedding, sequence_url_length)
        
        query_embedding = sequence_query
        query_seq_output = self.query_seq_encoder(query_embedding, sequence_query_length)

        # Concatenate user_features, sku_seq_output, url_seq_output, and query_seq_output
        combined_feat = torch.cat(
            [
                user_features, 
                sku_seq_output, 
                url_seq_output, 
                query_seq_output
            ], 
            dim=-1
        )
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


class MLPProjector(nn.Module):
    """
    A small neural network projection head that maps representations
    to the space where contrastive loss is applied.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(EMBEDDING_DIM, MLP_PROJECTION_DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
        item_stat_feat_dim: int,
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
            item_stat_feat_dim=item_stat_feat_dim,
        )
        self.mlp_projector = MLPProjector()
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
        user_rep_aug1 = self.mlp_projector(user_rep_aug1)
        user_rep_aug2 = self.mlp_projector(user_rep_aug2)
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
        x = val_batch
        self.forward(x)
        # preds = (
        #     self.task_net(user_rep) for self.task_net in self.task_nets
        # )
        # loss = self.loss_fn(preds, y)
        # self.log("val_loss", loss, prog_bar=True, logger=True)

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
