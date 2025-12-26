"""
SAM Fine-tuning Script for Beard Segmentation
Optimized for Apple M3 Mac with MPS (Metal Performance Shaders) acceleration.

Strategy:
- Freeze image encoder (pretrained, computationally expensive)
- Fine-tune prompt encoder + mask decoder (lightweight, domain-specific)
- Use mixed precision where supported
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything"))

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

from dataset import BeardCOCODataset, collate_fn


def get_device() -> torch.device:
    """Get the best available device, prioritizing MPS for M3 Mac."""
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Metal Performance Shaders) - Apple Silicon acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✓ Using CUDA")
        return torch.device("cuda")
    else:
        print("⚠ Using CPU - training will be slow")
        return torch.device("cpu")


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Focal + IoU prediction loss."""
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        bce_weight: float = 1.0,
        iou_weight: float = 1.0,
    ):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_iou: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_masks: Predicted masks [B, 1, H, W]
            gt_masks: Ground truth masks [B, H, W]
            pred_iou: Predicted IoU scores [B, 1]
        """
        # Resize gt_masks to match pred_masks if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode='nearest'
            ).squeeze(1)
        
        gt_masks = gt_masks.unsqueeze(1) if gt_masks.dim() == 3 else gt_masks
        
        # Calculate individual losses
        dice = self.dice_loss(pred_masks, gt_masks)
        focal = self.focal_loss(pred_masks, gt_masks)
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        
        # Calculate actual IoU for IoU prediction loss
        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
        actual_iou = intersection / (union + 1e-6)
        
        iou_loss = F.mse_loss(pred_iou.squeeze(-1), actual_iou)
        
        # Combined loss
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.bce_weight * bce +
            self.iou_weight * iou_loss
        )
        
        loss_dict = {
            'dice': dice.item(),
            'focal': focal.item(),
            'bce': bce.item(),
            'iou': iou_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict


class SAMTrainer:
    """
    Trainer for fine-tuning SAM on beard segmentation.
    """
    
    def __init__(
        self,
        model: Sam,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False,
    ):
        self.model = model
        self.device = device
        self.freeze_image_encoder = freeze_image_encoder
        
        # Move model to device
        self.model.to(device)
        
        # Freeze image encoder (keeps pretrained features, saves memory)
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            print("✓ Image encoder frozen")
        
        # Optionally freeze prompt encoder
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
            print("✓ Prompt encoder frozen")
        
        # Setup optimizer for trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = CombinedLoss()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
    
    def _prepare_inputs(self, batch: Dict) -> Tuple:
        """Prepare batch inputs for the model."""
        images = batch['image'].to(self.device)
        gt_masks = batch['gt_mask'].to(self.device)
        
        # Process batch through image encoder (with no_grad if frozen)
        if self.freeze_image_encoder:
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(self.model.preprocess(images))
        else:
            image_embeddings = self.model.image_encoder(self.model.preprocess(images))
        
        return images, image_embeddings, gt_masks, batch
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        # Keep image encoder in eval mode if frozen
        if self.freeze_image_encoder:
            self.model.image_encoder.eval()
        
        total_losses = {'dice': 0, 'focal': 0, 'bce': 0, 'iou': 0, 'total': 0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            images, image_embeddings, gt_masks, batch_data = self._prepare_inputs(batch)
            
            batch_size = images.shape[0]
            all_masks = []
            all_ious = []
            
            # Process each sample in batch (SAM processes one at a time)
            for i in range(batch_size):
                # Get prompts for this sample
                point_coords = batch_data['point_coords'][i].unsqueeze(0).to(self.device)
                point_labels = batch_data['point_labels'][i].unsqueeze(0).to(self.device)
                
                # Encode prompts
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                
                # Predict mask
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings[i:i+1],
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                all_masks.append(low_res_masks)
                all_ious.append(iou_predictions)
            
            # Stack predictions
            pred_masks = torch.cat(all_masks, dim=0)  # [B, 1, H, W]
            pred_ious = torch.cat(all_ious, dim=0)    # [B, 1]
            
            # Calculate loss
            loss, loss_dict = self.criterion(pred_masks, gt_masks, pred_ious)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] += v
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'dice': f"{loss_dict['dice']:.4f}",
            })
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_losses = {'dice': 0, 'focal': 0, 'bce': 0, 'iou': 0, 'total': 0}
        total_iou = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            images, image_embeddings, gt_masks, batch_data = self._prepare_inputs(batch)
            
            batch_size = images.shape[0]
            all_masks = []
            all_ious = []
            
            for i in range(batch_size):
                point_coords = batch_data['point_coords'][i].unsqueeze(0).to(self.device)
                point_labels = batch_data['point_labels'][i].unsqueeze(0).to(self.device)
                
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=image_embeddings[i:i+1],
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                all_masks.append(low_res_masks)
                all_ious.append(iou_predictions)
            
            pred_masks = torch.cat(all_masks, dim=0)
            pred_ious = torch.cat(all_ious, dim=0)
            
            loss, loss_dict = self.criterion(pred_masks, gt_masks, pred_ious)
            
            # Calculate actual IoU
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            if pred_binary.shape[-2:] != gt_masks.shape[-2:]:
                gt_resized = F.interpolate(
                    gt_masks.unsqueeze(1).float(),
                    size=pred_binary.shape[-2:],
                    mode='nearest'
                )
            else:
                gt_resized = gt_masks.unsqueeze(1)
            
            intersection = (pred_binary * gt_resized).sum()
            union = pred_binary.sum() + gt_resized.sum() - intersection
            batch_iou = (intersection / (union + 1e-6)).item()
            total_iou += batch_iou
            
            for k, v in loss_dict.items():
                total_losses[k] += v
            num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['mean_iou'] = total_iou / num_batches
        
        return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM for beard segmentation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "beard-dataset.v46i.coco-segmentation"),
        help="Path to beard dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint (downloads vit_b if not provided)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type (vit_b recommended for M3 Mac)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (keep low for MPS)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=3, help="Number of point prompts")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "checkpoints"),
        help="Output directory for checkpoints"
    )
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = BeardCOCODataset(
        args.data_dir,
        split="train",
        num_points=args.num_points,
    )
    val_dataset = BeardCOCODataset(
        args.data_dir,
        split="valid",
        num_points=args.num_points,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # MPS works best with num_workers=0
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    
    # Load model
    print("\n=== Loading SAM Model ===")
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    else:
        print(f"Loading model without checkpoint (requires downloading weights)")
        print(f"Download {args.model_type} weights from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print(f"Then run: python train_sam.py --checkpoint path/to/sam_vit_b_01ec64.pth")
        model = sam_model_registry[args.model_type]()
    
    # Create trainer
    print("\n=== Setting up Trainer ===")
    trainer = SAMTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        freeze_image_encoder=True,  # Always freeze for efficiency
        freeze_prompt_encoder=False,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(trainer.optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_val_iou = 0
    training_history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_losses = trainer.validate(val_loader)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_losses['total']:.4f}, Dice: {train_losses['dice']:.4f}")
        print(f"  Val   - Loss: {val_losses['total']:.4f}, Dice: {val_losses['dice']:.4f}, IoU: {val_losses['mean_iou']:.4f}")
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'lr': scheduler.get_last_lr()[0],
        })
        
        # Save best model
        if val_losses['mean_iou'] > best_val_iou:
            best_val_iou = val_losses['mean_iou']
            best_path = output_dir / "sam_beard_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_iou': best_val_iou,
                'model_type': args.model_type,
            }, best_path)
            print(f"  ✓ Saved best model (IoU: {best_val_iou:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"sam_beard_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_iou': val_losses['mean_iou'],
                'model_type': args.model_type,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()



