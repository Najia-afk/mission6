# Project: Product Classification Engine - Feasibility Study
## Multimodal Learning for E-Commerce Product Categorization

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-purple.svg)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-red.svg)](https://www.tensorflow.org/)

###  Project Context
This project evaluates the feasibility of automatic product classification using multimodal machine learning (text + images) for an e-commerce marketplace. The goal is to develop a production-ready classification engine that combines textual descriptions and product images to automatically assign products to categories.

The analysis progresses from exploratory data analysis (EDA) through transfer learning to advanced production-ready improvements including multimodal fusion, model interpretability, and experiment tracking.

###  Business & Technical Objectives
- **Feasibility Assessment**: Determine if text and/or image-based classification achieves acceptable accuracy for production use.
- **Architecture Validation**: Compare different feature extraction methods (TF-IDF, Word2Vec, BERT, USE, VGG16, CLIP).
- **Transfer Learning**: Leverage pre-trained models to minimize training time and data requirements.
- **Production Readiness**: Implement advanced features including:
  - Multimodal fusion (text + image)
  - Model interpretability (Grad-CAM visualizations)
  - Reproducibility (multi-seed training)
  - Experiment tracking (MLflow)
  - Alternative architectures (ResNet, EfficientNet, InceptionV3)

###  Technical Architecture
The project follows an 8-section structured workflow:

**Section 1-3: Exploratory Analysis**
- Data overview and class balance analysis
- Text preprocessing (lemmatization, stemming)
- Basic NLP encoding (Bag of Words, TF-IDF)

**Section 4: Advanced NLP**
- Word embeddings: Word2Vec, BERT, Universal Sentence Encoder (USE)
- Dimensionality reduction (PCA, t-SNE)
- Clustering quality assessment (Silhouette, ARI)

**Section 5: Image Processing & Deep Features**
- Classical features: SIFT, LBP, GLCM, Gabor filters
- Deep features: VGG16 (ImageNet pre-trained)
- Vision-language features: CLIP (SWIFT)
- Category-based evaluation

**Section 6: Unsupervised Transfer Learning**
- VGG16 feature extraction without supervision
- Clustering validation against true categories

**Section 7: Supervised Transfer Learning**
- Fine-tuned VGG16 classifier
- Training with and without data augmentation
- Confusion matrix and prediction visualization

**Section 8: Production-Ready Improvements**
1. **Enhanced Metrics**: Per-class F1, macro/micro averaging
2. **Grad-CAM**: Visual model interpretability with Plotly interactive heatmaps
3. **Multi-Seed Training**: Reproducibility & confidence intervals
4. **Alternative Backbones**: Architecture diversity (ResNet, EfficientNet, InceptionV3)
5. **Multimodal Fusion**: Late fusion of text (USE) + image (VGG16) features
6. **MLflow Tracking**: Experiment logging and model registry
7. **Summary**: Implementation checklist and next steps

---

###  Quick Start (Docker)

The environment includes Jupyter for notebooks and MLflow for experiment tracking.

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
```bash
cd mission6
docker-compose up --build
```

This starts:
- **Jupyter Container** (mission6_jupyter) on port 8886
- **MLflow Container** (mission6_mlflow) on port 5006

#### 3. Access the Services
- **Jupyter Notebook**: [http://localhost:8886](http://localhost:8886)
  - Open `mission6.ipynb` to run the complete analysis
- **MLflow UI**: [http://localhost:5006](http://localhost:5006)
  - View experiments, runs, metrics, and registered models

#### 4. Stop the Services
```bash
docker-compose down
```

---

###  Project Structure
```text
mission6/
├── mission6.ipynb              # Main analysis notebook (8 sections)
├── mission6.html               # Exported notebook results
├── docker-compose.yml          # Multi-container orchestration
├── Dockerfile                  # Python 3.10 environment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── dataset/
│   └── Flipkart/              # Product images and metadata
│       ├── flipkart*.csv      # Product catalog CSV files
│       └── Images/            # Product image files
├── src/
│   ├── classes/               # Core analysis classes
│   │   ├── advanced_embeddings.py         # Word2Vec, BERT, USE embeddings
│   │   ├── analyze_category_tree.py       # Category hierarchy analysis
│   │   ├── analyze_value_specifications.py # Value distribution analysis
│   │   ├── encode_text.py                  # TF-IDF and Bag-of-Words
│   │   ├── enhanced_metrics.py             # Per-class F1, macro/micro metrics
│   │   ├── grad_cam.py                     # Grad-CAM visualization
│   │   ├── image_processor.py              # Image preprocessing and features
│   │   ├── mlflow_tracker.py               # MLflow experiment tracking
│   │   ├── multimodal_fusion.py            # Late fusion classifier
│   │   ├── multi_seed_trainer.py           # Multi-seed reproducibility
│   │   ├── preprocess_text.py              # Text cleaning and preprocessing
│   │   ├── reduce_dimensions.py            # PCA, t-SNE, clustering
│   │   ├── swift_extractor.py              # CLIP-based feature extraction
│   │   ├── transfer_learning_classifier.py        # VGG16 supervised training
│   │   ├── transfer_learning_classifier_unsupervised.py # Unsupervised extraction
│   │   └── vgg16_extractor.py              # VGG16 deep feature extraction
│   └── scripts/                # Analysis and visualization utilities
├── mlruns/                     # MLflow experiment tracking database
└── models/                     # Saved model checkpoints
```

---

###  Key Findings & Insights

#### Text Analysis (Sections 3-4)
- **TF-IDF vs Word2Vec**: TF-IDF provides strong baseline; embeddings (BERT, USE) add semantic understanding.
- **Universal Sentence Encoder (USE)**: Best balance of semantic representation and computational efficiency.
- **Silhouette Analysis**: Product categories show meaningful separation in embedding space (silhouette scores 0.3-0.5).

#### Image Analysis (Sections 5-6)
- **Classical vs Deep Features**: Deep features (VGG16, CLIP) vastly outperform classical features (SIFT, LBP).
- **VGG16 Performance**: Effective transfer learning with frozen backbone; ARI scores 0.35-0.45 on unsupervised clustering.
- **CLIP Advantage**: Vision-language pre-training provides semantic alignment superior to VGG16 alone.

#### Transfer Learning (Section 7)
- **Base Model**: VGG16 with data augmentation achieves 65-75% accuracy on test set.
- **Augmentation Impact**: Data augmentation reduces overfitting gap and improves generalization.
- **Per-Class Performance**: Macro F1 scores indicate balanced performance across categories.

#### Multimodal Fusion (Section 8.5)
- **Fusion Improvement**: Late fusion (text + image) typically improves accuracy by 2-5% over single modality.
- **Complementary Information**: Text captures semantic category intent; images capture visual patterns.
- **Architecture Validation**: Different classifier depths (2-4 dense layers) show consistent improvement with fusion.

#### Production Readiness (Section 8)
- **Reproducibility**: Multi-seed training (3+ seeds) confirms model stability (std < 0.02 on test accuracy).
- **Interpretability**: Grad-CAM + Plotly visualizations show meaningful activation patterns in VGG16 layers with interactive heatmap overlays.
- **Tracking**: MLflow integration enables comprehensive experiment management and model registry.

---

###  Data & Model Specifications

#### Input Data
- **Product Images**: 224×224 RGB (resized from variable resolutions)
- **Product Text**: Titles and descriptions (cleaned, lemmatized, max 256 tokens)
- **Labels**: Multi-class product categories (7-20 categories typical)
- **Training Set**: ~1,050 images (sampled across categories)
- **Split**: 60% train / 15% validation / 25% test

#### Feature Dimensions
- **Text Features**:
  - TF-IDF: 500-2000 features
  - USE Embeddings: 512-dimensional vectors
  - BERT: 768-dimensional vectors
- **Image Features**:
  - VGG16 (block5_pool): 25,088 → 150-256 PCA dimensions
  - CLIP: 512-dimensional vectors
- **Multimodal Fusion**: USE (512) + VGG16-PCA (256) = 768 concatenated dimensions

#### Model Architecture
```
Transfer Learning Classifier:
  Input: (224, 224, 3) images
  ├─ VGG16 (frozen, ImageNet weights)
  ├─ GlobalAveragePooling2D
  ├─ Dense(1024, ReLU) + Dropout(0.5)
  ├─ Dense(512, ReLU) + Dropout(0.3)
  └─ Dense(num_classes, Softmax)
  
Multimodal Fusion:
  Input: Concatenated [USE_text (512) + VGG16_image (256)]
  ├─ Dense(512, ReLU) + Dropout(0.5)
  ├─ Dense(256, ReLU) + Dropout(0.3)
  └─ Dense(num_classes, Softmax)
```

---

###  Installation & Dependencies

#### Python Packages (Auto-installed via Docker)
```
tensorflow==2.13.0          # Deep learning framework
keras==2.13.0               # High-level neural networks API
scikit-learn==1.3.0         # Machine learning tools
numpy, pandas, scipy        # Numerical computing
matplotlib, plotly          # Visualization
opencv-python               # Image processing
nltk, spacy                 # NLP tools
gensim                      # Word2Vec embeddings
transformers                # BERT, CLIP models
sentence-transformers       # Universal Sentence Encoder
mlflow==2.7.0               # Experiment tracking
evidently==0.4.0            # Data drift detection
shap                        # Model interpretability
```

---

###  Execution Instructions

#### 1. Build and Start Docker Containers
```bash
cd mission6
docker-compose up --build
```

#### 2. Open Jupyter Notebook
```
Visit: http://localhost:8886
Password: (check logs or container config)
Open: mission6.ipynb
```

#### 3. Run Sections Sequentially
- **Sections 1-3**: ~5-10 minutes (EDA, text preprocessing)
- **Section 4**: ~15-20 minutes (Word embeddings, clustering)
- **Section 5**: ~30-40 minutes (Image features, VGG16, CLIP)
- **Section 6**: ~10-15 minutes (Unsupervised transfer learning)
- **Section 7**: ~20-30 minutes (Supervised transfer learning)
- **Section 8**: ~10-15 minutes (Production improvements, MLflow logging)

**Total Runtime**: ~90-140 minutes for full execution

#### 4. View MLflow Experiments
```
Visit: http://localhost:5006
- View all logged runs and metrics
- Compare model architectures
- Register best model to model registry
```

#### 5. Export Results
```bash
# Convert notebook to HTML
jupyter nbconvert --to html --execute mission6.ipynb

# Or use Docker:
docker exec mission6_jupyter jupyter nbconvert --to html --execute mission6.ipynb
```

---

###  Key Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Base VGG16 Accuracy** | 65-75% | Single image modality |
| **Macro F1** | 0.62-0.72 | Per-class F1 average |
| **Weighted F1** | 0.68-0.78 | Class-weighted F1 |
| **Multimodal Fusion Accuracy** | 67-78% | +2-5% improvement with text |
| **Multi-Seed Stability** | ±0.02 | Std dev across 3+ seeds |
| **Grad-CAM + Plotly** | Interactive heatmaps | Layer activation overlays on images |

---

###  Troubleshooting

#### Docker Issues
```bash
# Force rebuild without cache
docker-compose up --build --no-cache

# View container logs
docker logs mission6_jupyter
docker logs mission6_mlflow

# Clean up and restart
docker-compose down -v
docker-compose up --build
```

#### Memory Issues
- Reduce batch size in Section 8.3 from 32 to 16 or 8
- Limit multi-seed training to 2 seeds instead of 3
- Process fewer images in Section 5 (e.g., 500 instead of 1050)

#### Image Loading Issues
- Ensure `dataset/Flipkart/Images/` directory exists and contains .jpg files
- Run image preprocessing cell to auto-generate sample images if missing

#### MLflow Connection
```bash
# Ensure both containers are running
docker-compose ps

# Check MLflow is accessible
curl http://localhost:5006/health
```

---

###  Next Steps & Recommendations

**Short-term Enhancements**
1. ✅ Implement A/B testing framework for online evaluation
2. ✅ Add SHAP for model-agnostic local interpretability
3. ✅ Deploy API with FastAPI/Flask for real-time predictions
4. ✅ Integrate data drift monitoring (Evidently)

**Production Deployment**
1. Register best model in MLflow Model Registry
2. Create production Docker image with inference API
3. Set up monitoring and alerting for prediction drift
4. Implement continuous retraining pipeline

**Advanced Improvements**
1. Fine-tune CLIP backbone on domain-specific data
2. Implement attention mechanisms for interpretability
3. Add temporal tracking for category evolution
4. Develop ensemble methods (voting/stacking)

---

###  References & Resources

- **VGG16**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **BERT**: [Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Universal Sentence Encoder**: [Multilingual, Multitask Text Representations](https://arxiv.org/abs/1803.11175)
- **CLIP**: [Learning Transferable Models for Multimodal Task](https://arxiv.org/abs/2103.14030)
- **Grad-CAM**: [Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02055)
- **MLflow**: [Experiment Tracking & Model Registry](https://mlflow.org/docs/latest/)

---

*This project demonstrates end-to-end machine learning engineering: from exploratory data analysis through transfer learning to production-ready features including multimodal fusion, interpretability, and experiment tracking.*
