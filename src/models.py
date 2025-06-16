"""
Neural Network Models for Fraud Detection
Contains all model architectures including multi-modal and ensemble components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import DistilBertModel
from .constants import *


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = DEFAULT_FOCAL_ALPHA, gamma: float = DEFAULT_FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TabularNetwork(nn.Module):
    """Advanced tabular network with attention mechanism"""
    
    def __init__(self, input_dim: int, output_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE)
        )
        
        self.attention = nn.MultiheadAttention(
            128, num_heads=DEFAULT_NUM_ATTENTION_HEADS, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(128)
        
        self.feature_layer = nn.Sequential(
            nn.Linear(128, DEFAULT_EMBEDDING_DIM),
            nn.BatchNorm1d(DEFAULT_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_projection = nn.Linear(DEFAULT_EMBEDDING_DIM, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_transformer(x)
        x_sequence = x.unsqueeze(1)
        
        attention_output, _ = self.attention(x_sequence, x_sequence, x_sequence)
        x = self.attention_norm(x + attention_output.squeeze(1))
        
        x = self.feature_layer(x)
        return self.output_projection(x)


class GraphNetwork(nn.Module):
    """Graph neural network for entity relationships"""
    
    def __init__(self, input_dim: int, output_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        hidden_dim = DEFAULT_HIDDEN_DIM
        num_layers = 2
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.gnn_layers.append(SAGEConv(hidden_dim, output_dim))
        
        self.gat_layer = GATConv(
            output_dim, output_dim, heads=4, concat=False
        )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None) -> torch.Tensor:
        for gnn_layer, batch_norm in zip(self.gnn_layers, self.batch_norms):
            x = gnn_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.gat_layer(x, edge_index)
        x = F.relu(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class TextNetwork(nn.Module):
    """Text network using DistilBERT"""
    
    def __init__(self, output_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        try:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            bert_hidden_size = self.bert.config.hidden_size
            
            # Freeze BERT parameters for efficiency
            for param in self.bert.parameters():
                param.requires_grad = False
            
            self.text_projector = nn.Sequential(
                nn.Linear(bert_hidden_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(DEFAULT_DROPOUT_RATE),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, output_dim)
            )
            
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=bert_hidden_size, num_heads=8, batch_first=True
            )
            self.has_bert = True
            
        except Exception:
            self.has_bert = False
            self.mock_projection = nn.Linear(128, output_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.has_bert:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = bert_outputs.last_hidden_state
            
            pooled_output, _ = self.attention_pooling(
                hidden_states, hidden_states, hidden_states,
                key_padding_mask=~attention_mask.bool()
            )
            
            cls_output = pooled_output[:, 0, :]
            return self.text_projector(cls_output)
        else:
            batch_size = input_ids.size(0)
            return torch.randn(batch_size, DEFAULT_EMBEDDING_DIM, device=input_ids.device)


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        latent_dim = DEFAULT_LATENT_DIM
        hidden_dims = [128, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> tuple:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = F.mse_loss(reconstructed, x, reduction='none').mean(dim=1)
            return error


class MultiModalFusionLayer(nn.Module):
    """Multi-modal fusion with cross-attention"""
    
    def __init__(self):
        super().__init__()
        fusion_dim = DEFAULT_FUSION_DIM
        embedding_dim = DEFAULT_EMBEDDING_DIM
        
        self.tabular_projection = nn.Linear(embedding_dim, fusion_dim)
        self.graph_projection = nn.Linear(embedding_dim, fusion_dim)
        self.text_projection = nn.Linear(embedding_dim, fusion_dim)
        self.anomaly_projection = nn.Linear(1, fusion_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=DEFAULT_NUM_ATTENTION_HEADS,
            batch_first=True, 
            dropout=0.1
        )
        
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 4),
            nn.Softmax(dim=1)
        )
        
        self.fusion_layers = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, tab_embedding: torch.Tensor, graph_embedding: torch.Tensor,
                text_embedding: torch.Tensor, anomaly_score: torch.Tensor) -> tuple:
        batch_size = tab_embedding.size(0)
        
        # Project embeddings to common dimension
        tab_proj = self.tabular_projection(tab_embedding)
        graph_proj = self.graph_projection(graph_embedding)
        text_proj = self.text_projection(text_embedding)
        anomaly_proj = self.anomaly_projection(anomaly_score)
        
        # Stack modalities for attention
        modalities = torch.stack([tab_proj, graph_proj, text_proj, anomaly_proj], dim=1)
        
        # Apply cross-attention
        attended_modalities, attention_weights = self.cross_attention(
            modalities, modalities, modalities
        )
        
        # Residual connection
        attended_modalities = attended_modalities + modalities
        
        # Modality importance weighting
        concatenated = attended_modalities.reshape(batch_size, -1)
        importance_weights = self.modality_gate(concatenated)
        
        # Apply importance weights
        weighted_modalities = attended_modalities * importance_weights.unsqueeze(-1)
        final_features = weighted_modalities.reshape(batch_size, -1)
        
        # Final fusion and classification
        fused_features = self.fusion_layers(final_features)
        fraud_logits = self.classifier(fused_features)
        
        return fraud_logits.squeeze(-1), {
            'attention_weights': attention_weights,
            'importance_weights': importance_weights,
            'fused_features': fused_features
        }


def create_fraud_graph(batch_size: int = DEFAULT_BATCH_SIZE) -> Batch:
    """Create synthetic graph data for fraud detection"""
    graphs = []
    
    for i in range(batch_size):
        num_nodes = np.random.randint(5, 15)
        node_features = torch.randn(num_nodes, 32)
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Remove self-loops and duplicate edges
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        edge_index = torch.unique(edge_index, dim=1)
        
        graph_data = Data(x=node_features, edge_index=edge_index)
        graphs.append(graph_data)
    
    return Batch.from_data_list(graphs)


class MultiModalFraudDetector(pl.LightningModule):
    """Complete multi-modal fraud detection system"""
    
    def __init__(self, tabular_input_dim: int, use_focal_loss: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize networks
        self.tabular_network = TabularNetwork(tabular_input_dim)
        self.graph_network = GraphNetwork(32)  # Fixed graph input dimension
        self.text_network = TextNetwork()
        self.anomaly_autoencoder = AnomalyAutoencoder(tabular_input_dim)
        self.fusion_layer = MultiModalFusionLayer()
        
        # Loss function
        if use_focal_loss:
            self.loss_function = FocalLoss()
        else:
            self.loss_function = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task='binary')
        self.validation_accuracy = Accuracy(task='binary')
        self.validation_auroc = AUROC(task='binary')
        self.validation_precision = Precision(task='binary')
        self.validation_recall = Recall(task='binary')
        self.validation_f1 = F1Score(task='binary')
    
    def forward(self, tabular_data: torch.Tensor, graph_data=None, 
                text_ids=None, text_mask=None) -> tuple:
        batch_size = tabular_data.size(0)
        
        # Process tabular data
        tabular_embedding = self.tabular_network(tabular_data)
        
        # Process graph data
        if graph_data is None:
            graph_data = create_fraud_graph(batch_size)
            graph_data = graph_data.to(tabular_data.device)
        
        graph_embedding = self.graph_network(
            graph_data.x, graph_data.edge_index, graph_data.batch
        )
        
        # Ensure consistent batch size
        if graph_embedding.size(0) != batch_size:
            graph_embedding = graph_embedding[:batch_size]
        
        # Process text data (mock for now)
        if text_ids is None or text_mask is None:
            text_embedding = torch.randn(
                batch_size, DEFAULT_EMBEDDING_DIM, device=tabular_data.device
            )
        else:
            text_embedding = self.text_network(text_ids, text_mask)
        
        # Get anomaly scores
        anomaly_scores = self.anomaly_autoencoder.get_reconstruction_error(
            tabular_data
        ).unsqueeze(1)
        
        # Fusion and classification
        fraud_logits, auxiliary_info = self.fusion_layer(
            tabular_embedding, graph_embedding, text_embedding, anomaly_scores
        )
        
        return fraud_logits, auxiliary_info
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        tabular_data = batch['tab']
        labels = batch['label']
        
        fraud_logits, _ = self.forward(tabular_data)
        fraud_loss = self.loss_function(fraud_logits, labels)
        
        # Autoencoder reconstruction loss
        reconstructed, _ = self.anomaly_autoencoder(tabular_data)
        reconstruction_loss = F.mse_loss(reconstructed, tabular_data)
        
        total_loss = fraud_loss + 0.1 * reconstruction_loss
        
        # Calculate accuracy
        fraud_probabilities = torch.sigmoid(fraud_logits)
        predictions = fraud_probabilities > 0.5
        accuracy = self.train_accuracy(predictions, labels.int())
        
        # Logging
        self.log('train/fraud_loss', fraud_loss, prog_bar=True)
        self.log('train/reconstruction_loss', reconstruction_loss)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/accuracy', accuracy, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        tabular_data = batch['tab']
        labels = batch['label']
        
        fraud_logits, _ = self.forward(tabular_data)
        fraud_loss = self.loss_function(fraud_logits, labels)
        
        fraud_probabilities = torch.sigmoid(fraud_logits)
        predictions = fraud_probabilities > 0.5
        labels_int = labels.int()
        
        # Update metrics
        self.validation_accuracy(predictions, labels_int)
        self.validation_auroc(fraud_probabilities, labels_int)
        self.validation_precision(predictions, labels_int)
        self.validation_recall(predictions, labels_int)
        self.validation_f1(predictions, labels_int)
        
        # Logging
        self.log('validation/loss', fraud_loss, prog_bar=True)
        self.log('validation/accuracy', self.validation_accuracy, prog_bar=True)
        self.log('validation/auc', self.validation_auroc, prog_bar=True)
        self.log('validation/precision', self.validation_precision)
        self.log('validation/recall', self.validation_recall, prog_bar=True)
        self.log('validation/f1', self.validation_f1, prog_bar=True)
        
        return fraud_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'validation/auc'
        }