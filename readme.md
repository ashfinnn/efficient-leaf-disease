# CPU-Constrained Deep Learning for Tomato Disease Detection 🌱

This project evaluates **traditional, modern, and hybrid CNN architectures** — **ResNet-50**, **ConvNeXt-Tiny**, and **FastViT-T8** — for **tomato leaf disease classification** under **CPU-only constraints**. The study addresses the critical gap between GPU-accelerated research and real-world agricultural deployment in resource-limited settings.

**Paper**: *CPU-Constrained Deep Learning for Tomato Disease Detection: Traditional, Modern, and Hybrid CNN Comparison*  
**Authors**: **Obidur Rahman**, Lipon Chandra Das, Arnab Aich, Abu Saiman Md Taiham, Atif Ibna Latif  
**Affiliation**: Department of Mathematics, University of Chittagong, Bangladesh

---

## 📌 Key Highlights

- **Dataset**: PlantVillage tomato subset (16,012 images, 10 classes: 9 diseases + healthy)
- **Hardware**: Standard CPU-only system (AMD Ryzen 5 5600G) — no GPU required
- **Architectures Compared**:
  - **ResNet-50**: Classical residual CNN baseline
  - **ConvNeXt-Tiny**: Modernized convolutional network
  - **FastViT-T8**: Hybrid CNN-Transformer architecture
- **Training Protocol**: Two-phase transfer learning with ImageNet pre-training
- **Focus**: Practical deployment for smallholder farmers and agricultural extension services

---

## 🚀 Results Summary

| Model | Params (M) | Accuracy | Weighted F1 | Inference Time (s/image) | Best For |
|-------|-----------|----------|-------------|-------------------------|----------|
| **ConvNeXt-Tiny** | 29.0 | **99.88%** | **0.9988** | 0.051 | Maximum precision |
| **FastViT-T8** | 4.03 | 99.66% | 0.9966 | **0.022** | Real-time applications |
| **ResNet-50** | 25.6 | 97.69% | 0.9769 | 0.055 | Established baseline |

### Key Findings:
- ✅ **FastViT-T8** achieves optimal balance: 99.66% accuracy with **57% faster inference** than ConvNeXt-Tiny
- ✅ **ConvNeXt-Tiny** delivers highest accuracy, ideal for lab/centralized diagnostics
- ✅ Modern architectures significantly outperform ResNet-50 on both accuracy and efficiency
- ✅ All models handle class imbalance well (373–3,209 samples per class)

---

## 📂 Repository Structure

```
cpu-tomato-disease-detection/
├── datasets/               # PlantVillage tomato dataset
│   ├── Bacterial_Spot/
│   ├── Early_Blight/
│   ├── Late_Blight/
│   └── ... (10 classes total)
├── src/
│   ├── train.py           # Main training script (two-phase transfer learning)
│   ├── evaluate.py        # Model evaluation & metrics
│   └── utils.py           # Helper functions
├── trained_models/        # Saved checkpoints (.pth)
├── evaluation_results/    # Metrics JSON + confusion matrices
├── logs/                  # Training logs (CSV)
├── paper.pdf              # Full research paper
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ashfinnn/efficient-leaf-disease.git
cd efficient-leaf-disease
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0              # For modern architectures
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
numpy>=1.24.0
```

---

## 🖥️ Usage

### Training

Train any architecture with automatic two-phase transfer learning:

```bash
# Traditional CNN
python src/train.py --model resnet50 --batch-size 8 --epochs 20

# Modern CNN
python src/train.py --model convnext_tiny --batch-size 8 --epochs 20

# Hybrid Transformer
python src/train.py --model fastvit_t8 --batch-size 8 --epochs 20
```

**Training Features**:
- Phase 1 (Epochs 0-4): Frozen backbone, LR=1e-4
- Phase 2 (Epochs 5-19): Full fine-tuning, LR=5e-5
- Early stopping (patience=5)
- Gradient accumulation (effective batch size=32)
- Cosine annealing learning rate schedule

### Evaluation

```bash
python src/evaluate.py --model resnet50 --checkpoint trained_models/resnet50_best.pth
```

**Outputs**:
- Per-class precision, recall, F1-score
- Confusion matrix visualization
- Inference latency benchmark
- Overall accuracy metrics

---

## 📊 Performance Analysis

### Per-Class F1-Scores (Validation Set)

| Disease Class | ConvNeXt-Tiny | FastViT-T8 | ResNet-50 |
|---------------|---------------|------------|-----------|
| Bacterial Spot | 1.000 | 0.997 | 0.987 |
| Early Blight | 0.998 | 0.982 | **0.895** |
| Late Blight | 0.997 | 0.995 | 0.974 |
| Leaf Mold | 0.997 | 1.000 | 0.984 |
| Septoria Leaf Spot | 0.999 | 0.996 | 0.970 |
| Spider Mites | 0.999 | 1.000 | 0.984 |
| Target Spot | 0.996 | 0.995 | **0.955** |
| Yellow Curl Virus | 1.000 | 0.999 | 0.996 |
| Tomato Mosaic Virus | 1.000 | 1.000 | 0.960 |
| Healthy | 1.000 | 0.998 | 0.995 |

**Note**: ResNet-50 struggles most with Early Blight and Target Spot due to visual similarity with other diseases.

### Computational Efficiency

| Metric | FastViT-T8 | ConvNeXt-Tiny | ResNet-50 |
|--------|-----------|---------------|-----------|
| Time per Image | **0.022s** | 0.051s | 0.055s |
| Throughput | **~45 img/s** | ~20 img/s | ~18 img/s |
| 1000 Images | **22 seconds** | 51 seconds | 55 seconds |

---

## 🌾 Deployment Recommendations

### Field Applications (Real-time Diagnosis)
**Use FastViT-T8**:
- Fastest inference (0.022s/image)
- Excellent accuracy (99.66%)
- Suitable for mobile/edge devices
- Ideal for batch processing in extension services

### Laboratory/Centralized Systems
**Use ConvNeXt-Tiny**:
- Highest accuracy (99.88%)
- Minimal misclassifications
- Best for research and training datasets

### Legacy Systems
**Use ResNet-50**:
- Reliable baseline performance
- Well-documented architecture
- Suitable when modern libraries unavailable

---

## 🔬 Research Contributions

1. **CPU-Only Evaluation**: First comprehensive study comparing modern architectures under CPU constraints for agricultural AI
2. **Practical Guidance**: Clear deployment recommendations based on accuracy-efficiency trade-offs
3. **Class Imbalance Handling**: Demonstrates modern architectures' robustness (373–3,209 samples/class)
4. **Real-World Feasibility**: Proves high-accuracy disease detection without GPU dependency

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Single random seed (42) — multi-seed validation needed
- Controlled dataset conditions (PlantVillage)
- Potential data leakage (same-plant images in train/validation)
- Single CPU configuration tested

### Future Directions
1. **Field Validation**: Test on real agricultural images with variable lighting, occlusions, co-infections
2. **Statistical Robustness**: Multi-seed experiments with McNemar's test
3. **Plant-Level Splitting**: Eliminate data leakage by splitting at plant ID level
4. **Mobile Optimization**: Model quantization and compression for smartphone deployment
5. **Multi-Crop Extension**: Adapt framework for rice, wheat, potato diseases
6. **Ensemble Methods**: Combine predictions across checkpoints for improved generalization

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{rahman2025cpu,
  title={CPU-Constrained Deep Learning for Tomato Disease Detection: Traditional, Modern, and Hybrid CNN Comparison},
  author={Rahman, Obidur and Das, Lipon Chandra and Aich, Arnab and Taiham, Abu Saiman Md and Latif, Atif Ibna},
  journal={Department of Mathematics, University of Chittagong},
  year={2025}
}
```

---

## 🤝 Contact

**Corresponding Author**: Lipon Chandra Das  
**Email**: lipon@cu.ac.bd  
**Affiliation**: Department of Mathematics, University of Chittagong, Bangladesh

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Hughes & Salathe (2016)
- **TIMM Library**: PyTorch Image Models by Ross Wightman
- **University of Chittagong**: Department of Mathematics

---

## 🌍 Impact Statement

This research addresses critical barriers to AI adoption in agriculture by:
- **Reducing deployment costs** (no GPU required)
- **Improving accessibility** for smallholder farmers in developing regions
- **Enabling real-time diagnosis** with consumer-grade hardware
- **Supporting global food security** through timely disease detection

**Target Regions**: South Asia, Sub-Saharan Africa, and other resource-constrained agricultural communities.
