# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import math
import random
import copy
from tqdm import tqdm # For progress bars

# --- 0. Configuration / Hyperparameters ---
# Data Parameters
NUM_SAMPLES = 10000
TRAIN_RATIO = 0.7
SEQ_LENGTH = 128  # Example sequence length (e.g., depth points)
NUM_FEATURES = 8   # Example number of logging features (l)
NUM_CLASSES = 5    # Example number of lithology classes

# Model Parameters
EMBED_DIM = 128      # Transformer embedding dimension (r)
NUM_HEADS = 8        # Transformer multi-head attention heads
NUM_ENCODER_LAYERS = 6 # Transformer encoder layers
DIM_FEEDFORWARD = 512 # Transformer feedforward layer dimension
DROPOUT = 0.1
PROJECTION_DIM = 64 # Dimension for contrastive projection head

# Pre-training Parameters
PRETRAIN_EPOCHS = 20 # Reduced for faster demo, adjust as needed (e.g., 50-100)
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_LR = 1e-4
MU1 = 1.0  # Weight for L_D (mu_1 in Eq 7)
MU2 = 1.0  # Weight for L_Comparison (mu_2 in Eq 7)
CONTRASTIVE_K = 5 # Temporal steps for L_Comparison (K in Eq 3, 4)
AUG_Q = 5 # Max partitions for T_Y augmentation

# Fine-tuning Parameters
FINETUNE_EPOCHS = 15 # Reduced for faster demo, adjust as needed (e.g., 30-50)
FINETUNE_BATCH_SIZE = 128
FINETUNE_LR = 5e-5
LR_DECAY_STEP = 5 # Reduced step for demo
LR_DECAY_GAMMA = 0.5

# Other
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42 # For reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")

# --- 1. Data Augmentation (Section 2.1) ---

def basic_data_enrichment(x):
    """ T^X: Modifies signal amplitudes through scaling transformations. """
    # x shape: (batch, seq_len, features)
    scaling_factor = torch.randn(1, 1, x.size(-1), device=x.device) * 0.1 + 1.0 # Small random scaling per feature
    return x * scaling_factor

def advanced_data_diversification(x, q=AUG_Q):
    """ T^Y: Stochastic segmentation, sequence permutation, and noise injection. """
    # x shape: (batch, seq_len, features)
    x_aug = x.clone()
    batch_size, seq_len, _ = x.shape

    # 1. Stochastic Segmentation and Permutation (Simplified: Shuffle segments)
    num_partitions = random.randint(1, q)
    if num_partitions > 1 and seq_len >= num_partitions:
        seg_len = seq_len // num_partitions
        indices = list(range(num_partitions))
        random.shuffle(indices)

        permuted_x = torch.zeros_like(x_aug)
        current_pos = 0
        original_indices = []
        for p_idx in range(num_partitions):
             start = p_idx * seg_len
             end = (p_idx + 1) * seg_len if p_idx < num_partitions - 1 else seq_len
             original_indices.append((start, end))

        for i, p_idx in enumerate(indices):
            start, end = original_indices[p_idx]
            len_seg = end - start

            if current_pos + len_seg <= seq_len:
                 permuted_x[:, current_pos:current_pos+len_seg, :] = x_aug[:, start:end, :]
            else: # Handle potential length mismatch due to integer division
                 len_to_copy = seq_len - current_pos
                 if len_to_copy > 0:
                    permuted_x[:, current_pos:, :] = x_aug[:, start:start+len_to_copy, :]

            current_pos += len_seg
            if current_pos >= seq_len:
                break
        # Ensure the entire sequence is filled if segmentation was imperfect
        if current_pos < seq_len:
             remaining_len = seq_len - current_pos
             # Fill remaining with data from the start of the last segment perhaps?
             last_start, last_end = original_indices[indices[-1]]
             fill_data = x_aug[:, last_start : last_start + remaining_len, :]
             if fill_data.shape[1] == remaining_len: # Check if enough data exists
                 permuted_x[:, current_pos:, :] = fill_data
             # else: handle edge case where last segment is too short (less likely with integer division)

        x_aug = permuted_x


    # 2. Controlled Noise Injection
    noise = torch.randn_like(x_aug) * 0.05 # Adjust noise level as needed
    x_aug += noise

    return x_aug

# --- 2. Model Architecture ---

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding. """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Keep pe shape as (max_len, d_model) for easier broadcasting with batch_first=True
        # pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model) -> No, keep simple
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) due to batch_first=True
        # Add positional encoding to the input tensor x.
        # self.pe shape: (max_len, d_model) -> slice to (seq_len, d_model)
        # Unsqueeze to add batch dimension for broadcasting: (1, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """ Encoder based on Transformer (Section 2.2) """
    def __init__(self, input_dim, embed_dim, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, # Expect (batch, seq, feature)
            norm_first=True # Pre-Normalization as mentioned
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.input_proj.bias.data.zero_()

    def forward(self, src):
        # src shape: (batch, seq_len, input_dim)
        src = self.input_proj(src) * math.sqrt(self.embed_dim) # Project input features to embedding dim
        src = self.pos_encoder(src) # Apply positional encoding
        output = self.transformer_encoder(src) # (batch, seq_len, embed_dim)
        return output


class ProjectionHead(nn.Module):
    """ MLP head for projecting representations for contrastive loss. """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Add batchnorm for stability
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x can be (batch, input_dim) or (batch, seq_len, input_dim)
        # If sequential, apply linear layers across the feature dimension
        if x.dim() == 3:
            # Apply MLP to the pooled representation if needed, or handle sequence here
            # Assuming input is already pooled (batch, input_dim) for projection head
             x = self.fc1(x)
             # BatchNorm1d expects (batch, features) or (batch, features, seq_len)
             # If input is (batch, features), permute is not needed
             # If input were (batch, seq_len, features), need x.permute(0, 2, 1) for BN1d
             if x.dim() == 2: # (batch, hidden_dim)
                 x = self.bn1(x)
             x = self.relu(x)
             x = self.fc2(x)
        elif x.dim() == 2: # (batch, input_dim)
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.fc2(x)
        else:
            raise ValueError("ProjectionHead expects input dim 2 or 3")
        return x

class ContrastiveModel(nn.Module):
    """ Combines Encoder and Projection Heads for Pre-training (Section 2.1 & 2.3) """
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(encoder.embed_dim, encoder.embed_dim // 2, projection_dim)
        # Head for L_Comparison prediction (M_k in Eq 3, 4)
        # Predicts a vector used for comparison against future sequence embeddings
        self.temporal_prediction_head = ProjectionHead(encoder.embed_dim, encoder.embed_dim // 2, encoder.embed_dim) # Predicts vectors of embed_dim

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        s = self.encoder(x) # s shape: (batch, seq_len, embed_dim)

        # Get projections P for L_D (using mean pooling over sequence)
        pooled_s = s.mean(dim=1) # (batch, embed_dim)
        p = self.projection_head(pooled_s) # p shape: (batch, projection_dim)

        # Get context vector d for L_Comparison (using mean pooling)
        d = pooled_s # Use the same pooled representation as context (batch, embed_dim)

        # s represents the sequence of embeddings from the encoder
        # s shape: (batch, seq_len, embed_dim)

        return s, p, d

# --- 3. Contrastive Loss Functions (Section 2.3) ---

def compute_similarity_matrix(p_x, p_y):
    """ Computes cosine similarity matrix between two sets of projections. """
    p_x_norm = F.normalize(p_x, p=2, dim=1)
    p_y_norm = F.normalize(p_y, p=2, dim=1)
    sim_matrix = torch.mm(p_x_norm, p_y_norm.t()) # (batch_x, batch_y)
    return sim_matrix

def compute_l_d(p_x, p_y):
    """ Computes L_D loss (Eq 5, 6). Uses Barlow Twins style for stability. """
    # p_x, p_y shape: (batch, projection_dim)
    batch_size = p_x.size(0)
    if batch_size == 0: return torch.tensor(0.0, device=p_x.device)

    # Normalize projections along the batch dimension (Barlow Twins specific)
    p_x_norm = (p_x - p_x.mean(dim=0)) / (p_x.std(dim=0) + 1e-5)
    p_y_norm = (p_y - p_y.mean(dim=0)) / (p_y.std(dim=0) + 1e-5)

    # Cross-correlation matrix
    c = torch.mm(p_x_norm.T, p_y_norm) / batch_size # (projection_dim, projection_dim)

    # Loss calculation
    on_diag = torch.diagonal(c)
    loss_similar = ((1 - on_diag)**2).sum() # Maximize similarity along diagonal

    off_diag = c.fill_diagonal_(0)
    loss_dissimilar = (off_diag**2).sum() # Minimize correlation off-diagonal

    # Original paper might have weighting factors, simplified here
    l_d = loss_similar + loss_dissimilar
    return l_d


def compute_l_comparison(s_x, d_x, s_y, d_y, model, k_steps):
    """ Computes L_Comparison loss (Eq 3, 4) using InfoNCE. """
    # s_x, s_y shape: (batch, seq_len, embed_dim) - Encoder outputs
    # d_x, d_y shape: (batch, embed_dim) - Context vectors (pooled)
    # model: The ContrastiveModel instance
    # k_steps: Number of future steps K

    batch_size, seq_len, embed_dim = s_x.shape
    if seq_len <= k_steps:
        return torch.tensor(0.0, device=s_x.device)

    total_loss = 0.0
    num_valid_steps = 0

    # Predict future representations using the context vectors d_x and d_y
    # M_k(d^X) -> predicts a representation based on d_x
    predicted_features_x = model.temporal_prediction_head(d_x) # (batch, embed_dim)
    # M_k(d^Y) -> predicts a representation based on d_y
    predicted_features_y = model.temporal_prediction_head(d_y) # (batch, embed_dim)

    # Normalize predictions and sequence embeddings for InfoNCE
    pred_x_norm = F.normalize(predicted_features_x, dim=1)
    pred_y_norm = F.normalize(predicted_features_y, dim=1)
    s_x_norm = F.normalize(s_x, dim=2) # Normalize along feature dimension
    s_y_norm = F.normalize(s_y, dim=2)

    # Calculate loss for each step k from 1 to K
    for k in range(1, k_steps + 1):
        # Target representations: s_{t+k}^Y and s_{t+k}^X
        # We need to contrast the prediction against the *correct* future step s_{t+k}
        # and treat other steps/samples as negatives.

        # Let's contrast the prediction against the embedding at exactly step t+k
        # Need to select a reference time 't'. Let's use t=0 for simplicity.
        # Target index for step k is simply 'k'.
        if k >= seq_len: continue # Ensure target index is within bounds

        # Positive targets at step k (relative to start t=0)
        s_target_y_k = s_y_norm[:, k, :] # (batch, embed_dim) - Targets for pred_x
        s_target_x_k = s_x_norm[:, k, :] # (batch, embed_dim) - Targets for pred_y

        # Calculate logits for Eq 3: (M_k(d^X))^T * s_n^Y (at step k)
        # Positive similarity: dot product between pred_x and its corresponding s_target_y_k
        # Negative similarity: dot product between pred_x and other s_target_y_k in the batch
        logits_x_vs_y_k = torch.mm(pred_x_norm, s_target_y_k.t()) # (batch, batch)

        # Calculate logits for Eq 4: (M_k(d^Y))^T * s_n^X (at step k)
        logits_y_vs_x_k = torch.mm(pred_y_norm, s_target_x_k.t()) # (batch, batch)

        # InfoNCE Loss
        labels = torch.arange(batch_size, device=s_x.device)
        loss_comp_x_k = F.cross_entropy(logits_x_vs_y_k, labels)
        loss_comp_y_k = F.cross_entropy(logits_y_vs_x_k, labels)

        total_loss += (loss_comp_x_k + loss_comp_y_k)
        num_valid_steps += 1

    if num_valid_steps == 0:
        return torch.tensor(0.0, device=s_x.device)

    # Average loss over K steps
    l_comparison = total_loss / num_valid_steps
    return l_comparison

# --- 4. Supervised Model for Fine-tuning ---

class LithologyClassifier(nn.Module):
    """ Fine-tuning model: Pre-trained Encoder + Classification Head """
    def __init__(self, pretrained_encoder, num_classes):
        super().__init__()
        # Make a deep copy to avoid modifying the original pre-trained encoder during fine-tuning
        self.encoder = copy.deepcopy(pretrained_encoder)
        # Use the output of the encoder (e.g., mean pooling) for classification
        self.classifier = nn.Linear(pretrained_encoder.embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        features = self.encoder(x) # (batch, seq_len, embed_dim)
        # Pool features across sequence length dimension (e.g., mean pooling)
        pooled_features = features.mean(dim=1) # (batch, embed_dim)
        # Alternative: Use the [CLS] token if you added one during encoding

        logits = self.classifier(pooled_features) # (batch, num_classes)
        return logits

# --- 5. Dataset Definition ---

class WellLogDataset(Dataset):
    """ Custom Dataset for Well Logging Data """
    def __init__(self, data, labels=None, is_pretrain=False):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.is_pretrain = is_pretrain # If true, __getitem__ returns only data for contrastive learning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx] # Shape: (seq_len, features)

        if self.is_pretrain:
            # For pre-training, we only need the data sample itself
            return sample_data
        else:
            # For fine-tuning/testing, return data and label
            if self.labels is None:
                raise ValueError("Labels must be provided for fine-tuning/testing dataset")
            sample_label = self.labels[idx]
            return sample_data, sample_label

# --- 6. Training and Evaluation Functions ---

def pretrain_epoch(model, dataloader, optimizer, device, mu1, mu2, k_steps):
    """ Runs one epoch of contrastive pre-training. """
    model.train()
    total_loss = 0.0
    total_l_d = 0.0
    total_l_comp = 0.0

    progress_bar = tqdm(dataloader, desc="Pretrain Epoch", leave=False)
    for batch_data in progress_bar:
        batch_data = batch_data.to(device) # (batch, seq_len, features)

        # Generate two augmented views
        x_basic = basic_data_enrichment(batch_data) # T^X
        x_advanced = advanced_data_diversification(batch_data) # T^Y

        optimizer.zero_grad()

        # Forward pass for both views
        s_x, p_x, d_x = model(x_basic)   # s^X, P^X, d^X
        s_y, p_y, d_y = model(x_advanced) # s^Y, P^Y, d^Y

        # Calculate losses
        l_d = compute_l_d(p_x, p_y)
        l_comparison = compute_l_comparison(s_x, d_x, s_y, d_y, model, k_steps)

        # Combine losses (Eq 7)
        loss = mu1 * l_d + mu2 * l_comparison

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l_d += l_d.item()
        total_l_comp += l_comparison.item()
        progress_bar.set_postfix(loss=loss.item(), ld=l_d.item(), lc=l_comparison.item())

    avg_loss = total_loss / len(dataloader)
    avg_l_d = total_l_d / len(dataloader)
    avg_l_comp = total_l_comp / len(dataloader)
    return avg_loss, avg_l_d, avg_l_comp

def finetune_epoch(model, dataloader, optimizer, criterion, device):
    """ Runs one epoch of supervised fine-tuning. """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Finetune Epoch", leave=False)
    for batch_data, batch_labels in progress_bar:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_data) # (batch, num_classes)
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions += (predicted_labels == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
        progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples if total_samples > 0 else 0)


    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """ Evaluates the model on the test set. """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_data, batch_labels in progress_bar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Forward pass
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted_labels = torch.max(logits, 1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples if total_samples > 0 else 0)


    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy
# --- 7. Main Execution ---

if __name__ == "__main__":
    print("--- Generating Dummy Data ---")
    # Replace with your actual data loading logic
    # Data shape: (num_samples, seq_length, num_features)
    dummy_data = np.random.randn(NUM_SAMPLES, SEQ_LENGTH, NUM_FEATURES).astype(np.float32)
    # Labels shape: (num_samples,) - integer labels from 0 to NUM_CLASSES-1
    dummy_labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES).astype(np.int64)
    print(f"Data shape: {dummy_data.shape}")
    print(f"Labels shape: {dummy_labels.shape}")

    print("\n--- Creating Datasets and DataLoaders ---")
    # Full dataset, includes label info for splitting if needed based on Dataset object
    # full_dataset = WellLogDataset(dummy_data, dummy_labels, is_pretrain=False) # If splitting based on Dataset object

    # Split dataset indices
    num_train = int(TRAIN_RATIO * NUM_SAMPLES)
    num_test = NUM_SAMPLES - num_train
    # Create and shuffle indices (if random split is desired)
    indices = np.arange(NUM_SAMPLES)
    np.random.shuffle(indices) # Ensure randomness
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # Create datasets for each phase
    # Pre-training uses only training data, without labels during loading
    pretrain_dataset = WellLogDataset(dummy_data[train_indices], is_pretrain=True)
    # Fine-tuning uses training data with labels
    finetune_train_dataset = WellLogDataset(dummy_data[train_indices], dummy_labels[train_indices], is_pretrain=False)
    # Testing uses test data with labels
    test_dataset = WellLogDataset(dummy_data[test_indices], dummy_labels[test_indices], is_pretrain=False)

    # Create DataLoaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False) # Set num_workers > 0 for parallel loading if needed
    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if DEVICE == 'cuda' else False)

    print(f"Pre-training samples: {len(pretrain_dataset)}")
    print(f"Fine-tuning samples: {len(finetune_train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    print("\n--- Initializing Models ---")
    # Base Encoder
    base_encoder = TransformerEncoder(
        input_dim=NUM_FEATURES,
        embed_dim=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    # Contrastive Model for Pre-training
    contrastive_model = ContrastiveModel(
        encoder=base_encoder,
        projection_dim=PROJECTION_DIM
    ).to(DEVICE)

    print("\n--- Starting Contrastive Pre-training ---")
    pretrain_optimizer = optim.AdamW(contrastive_model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        avg_loss, avg_l_d, avg_l_comp = pretrain_epoch(
            contrastive_model, pretrain_loader, pretrain_optimizer, DEVICE, MU1, MU2, CONTRASTIVE_K
        )
        print(f"Pretrain Epoch {epoch}/{PRETRAIN_EPOCHS} - Avg Loss: {avg_loss:.4f}, Avg L_D: {avg_l_d:.4f}, Avg L_Comp: {avg_l_comp:.4f}")

    # Extract the pre-trained encoder weights (p_Trained) - Note: contrastive_model.encoder now holds the trained weights
    print("\n--- Pre-training Finished ---")

    # --- Fine-tuning Phase ---
    print("\n--- Initializing Classifier for Fine-tuning ---")
    # Create the classifier model, loading the pre-trained encoder
    # Note: Pass the trained contrastive_model.encoder instance directly
    # LithologyClassifier should use copy.deepcopy internally if the original encoder state needs preservation
    classifier_model = LithologyClassifier(
        pretrained_encoder=contrastive_model.encoder, # Pass the trained encoder instance
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # Optional: Freeze encoder layers initially for fine-tuning only the classifier head
    # for param in classifier_model.encoder.parameters():
    #     param.requires_grad = False

    # Optimize only parameters that require gradients (if layers are frozen)
    finetune_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, classifier_model.parameters()), lr=FINETUNE_LR)
    # finetune_optimizer = optim.AdamW(classifier_model.parameters(), lr=FINETUNE_LR) # Optimize all parameters
    finetune_criterion = nn.CrossEntropyLoss()
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(finetune_optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)


    print("\n--- Starting Supervised Fine-tuning ---")
    best_test_acc = 0.0
    best_model_path = "best_lithology_classifier_model.pth" # Define model save path

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        # Optional: Unfreeze encoder layers after a few epochs
        # if epoch == 5: # Example: unfreeze after 4 epochs
        #     print("Unfreezing encoder layers...")
        #     for param in classifier_model.encoder.parameters():
        #         param.requires_grad = True
        #     # Re-initialize optimizer to include encoder parameters if they were frozen
        #     # May need to adjust learning rate
        #     finetune_optimizer = optim.AdamW(classifier_model.parameters(), lr=FINETUNE_LR / 5) # Use smaller LR for the whole model
        #     scheduler = optim.lr_scheduler.StepLR(finetune_optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)


        train_loss, train_acc = finetune_epoch(
            classifier_model, finetune_train_loader, finetune_optimizer, finetune_criterion, DEVICE
        )
        print(f"Finetune Epoch {epoch}/{FINETUNE_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate on test set periodically
        test_loss, test_acc = evaluate(
            classifier_model, test_loader, finetune_criterion, DEVICE
        )
        print(f"Finetune Epoch {epoch}/{FINETUNE_EPOCHS} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save the model if it has the best test accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save the fine-tuned model's state dict (including encoder and classifier)
            torch.save(classifier_model.state_dict(), best_model_path)
            print(f"*** New best model saved with Test Accuracy: {best_test_acc:.4f} ***")

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} LR: {current_lr:.6f}")


    print("\n--- Fine-tuning Finished ---")
    print(f"Best Test Accuracy achieved during fine-tuning: {best_test_acc:.4f}")

    # --- Optional: Load the best model and perform final evaluation ---
    print("\n--- Evaluating the Best Saved Model ---")
    # Initialize a new model instance structure
    final_encoder = TransformerEncoder(
        input_dim=NUM_FEATURES, embed_dim=EMBED_DIM, nhead=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(DEVICE) # Create the base encoder structure first

    final_classifier_model = LithologyClassifier(
        pretrained_encoder=final_encoder, # Pass the base structure
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # Load the saved state dictionary
    try:
        final_classifier_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print("Best model loaded successfully.")

        # Evaluate the loaded model
        final_test_loss, final_test_acc = evaluate(
            final_classifier_model, test_loader, finetune_criterion, DEVICE
        )
        print(f"Final Evaluation on Test Set - Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.4f}")

    except FileNotFoundError:
        print(f"Could not find '{best_model_path}'. Skipping final evaluation.")
    except Exception as e:
        print(f"An error occurred while loading or evaluating the best model: {e}")

    print("\n--- Script Finished ---")
