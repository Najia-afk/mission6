
# This script defines the content for the updated Part 8 and Part 9 of the notebook.
# It is intended to be used as a reference for updating the notebook cells.

# --- CELL: 8.0 Setup ---
import os
# Disable oneDNN optimizations to avoid numerical differences
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
import importlib
import time

# Reload custom modules to ensure latest code is used
import src.classes.enhanced_metrics as em
import src.classes.grad_cam as gc
import src.classes.multi_seed_trainer as mst
import src.classes.transfer_learning_classifier as tlc
import src.classes.multimodal_analysis as ma
import src.classes.mlflow_tracker as mlt

importlib.reload(em)
importlib.reload(gc)
importlib.reload(mst)
importlib.reload(tlc)
importlib.reload(ma)
importlib.reload(mlt)

from src.classes.enhanced_metrics import EnhancedMetrics
from src.classes.grad_cam import GradCAMVisualizer
from src.classes.multi_seed_trainer import MultiSeedTrainer
from src.classes.transfer_learning_classifier import TransferLearningClassifier
from src.classes.multimodal_analysis import MultimodalAnalysis
from src.classes.mlflow_tracker import MLflowTracker

print("✅ Environment configured and modules loaded.")

# --- CELL: 8.1 Enhanced Metrics ---
# Calculate metrics using the EnhancedMetrics class
if 'classifier' in locals():
    print("📊 Calculating Enhanced Metrics...")
    metrics_analyzer = EnhancedMetrics(classifier)
    
    # Calculate and print metrics
    metrics_report = metrics_analyzer.calculate_metrics(
        classifier.X_test, 
        classifier.test_df['product_category'].values
    )
    
    # Visualization: F1 Score per Category (Bar Chart)
    fig_bar = px.bar(
        x=list(metrics_report['per_class_f1'].keys()),
        y=list(metrics_report['per_class_f1'].values()),
        labels={'x': 'Category', 'y': 'F1 Score'},
        title='Per-Class F1 Score',
        color=list(metrics_report['per_class_f1'].values()),
        color_continuous_scale='Viridis'
    )
    fig_bar.show()

    # Visualization: Class Distribution (Pie Chart)
    # Showing the support (number of samples) per category in Test Set
    test_counts = classifier.test_df['product_category'].value_counts().reset_index()
    test_counts.columns = ['product_category', 'count']
    
    fig_pie = px.pie(
        test_counts, 
        values='count', 
        names='product_category', 
        title='Test Set Class Distribution (Support)',
        hover_data=['count'],
        labels={'product_category':'Category'}
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.show()
else:
    print("⚠️ 'classifier' object not found. Please run Section 7 first.")

# --- CELL: 8.2 Model Interpretability (Grad-CAM) ---
if 'classifier' in locals():
    print("🔍 Generating Grad-CAM visualizations...")
    gradcam = GradCAMVisualizer(classifier)
    
    # Visualize for a few random test images
    # We use the last convolutional block of VGG16: 'block5_conv3'
    gradcam.visualize(
        classifier.test_df, 
        num_samples=3, 
        layer_name='block5_conv3' 
    )
else:
    print("⚠️ 'classifier' object not found.")

# --- CELL: 8.3 Reproducibility (Multi-Seed Training) ---
print("🔄 Starting Multi-Seed Training Analysis...")
# Initialize trainer
trainer = MultiSeedTrainer(
    base_model_name='VGG16',
    num_seeds=3
)

# Prepare data
if 'df_sampled' in locals():
    trainer.prepare_data(
        df=df_sampled,
        image_col='image_path',
        label_col='product_category'
    )
    
    # Run training
    seed_results = trainer.train_and_evaluate(epochs=5, batch_size=32)
    
    # Visualize stability
    trainer.plot_results()
else:
    print("⚠️ 'df_sampled' not found. Please ensure data is loaded.")

# --- CELL: 8.4 Architecture Comparison ---
print("🏗️ Starting Architecture Comparison (VGG16 vs EfficientNetB0 vs MobileNetV3)...")

models_to_compare = ['VGG16', 'EfficientNetB0', 'MobileNetV3Small']
comp_results = []

if 'df_sampled' in locals():
    for model_name in tqdm(models_to_compare, desc="Comparing Models"):
        print(f"\nTraining {model_name}...")
        
        # Initialize a temporary classifier for this architecture
        temp_clf = TransferLearningClassifier(
            input_shape=(224, 224, 3),
            base_model_name=model_name
        )
        
        temp_clf.prepare_data_from_dataframe(
            df=df_sampled,
            image_column='image_path',
            category_column='product_category',
            test_size=0.2,
            val_size=0.25
        )
        temp_clf.prepare_arrays_method()
        
        model = temp_clf.create_base_model()
        
        # Train briefly (5 epochs)
        temp_clf.train_model(
            model_name=f"{model_name}_comp",
            model=model,
            epochs=5,
            batch_size=32,
            patience=2
        )
        
        # Record metrics
        res = temp_clf.evaluation_results.get(f"{model_name}_comp", {})
        comp_results.append({
            'Model': model_name,
            'Accuracy': res.get('accuracy', 0),
            'Training Time (s)': res.get('training_time', 0),
            'Parameters': model.count_params()
        })

    # Visualize Comparison
    comp_df = pd.DataFrame(comp_results)
    
    # Accuracy vs Time Bubble Chart
    fig_eff = px.scatter(
        comp_df, 
        x='Training Time (s)', 
        y='Accuracy', 
        size='Parameters', 
        color='Model',
        title='Architecture Comparison: Accuracy vs Efficiency (Size = Params)',
        text='Model',
        hover_data=['Parameters']
    )
    fig_eff.update_traces(textposition='top center')
    fig_eff.show()
    
    print("\nComparison Results:")
    print(comp_df)
else:
    print("⚠️ 'df_sampled' not found.")

# --- CELL: 8.5 Multimodal Fusion ---
if 'classifier' in locals():
    print("🔗 Running Multimodal Fusion (Text + Image)...")
    multimodal = MultimodalAnalysis(classifier)
    
    # Evaluate fusion
    # Requires text data in test_df
    fusion_results = multimodal.evaluate_fusion(
        classifier.X_test,
        classifier.test_df['product_category'].values,
        classifier.test_df['description'].values
    )
else:
    print("⚠️ 'classifier' object not found.")

# --- CELL: 8.6 MLflow Tracking ---
print("📝 Logging experiments to MLflow...")
tracker = MLflowTracker(experiment_name="Mission6_Final_Run")

tracker.start_run(run_name="Production_Candidate")
tracker.log_params({
    'model': 'VGG16', 
    'fusion': True,
    'epochs': 5
})

# Log metrics if available
if 'metrics_report' in locals():
    tracker.log_metrics({
        'weighted_f1': metrics_report.get('weighted_f1', 0),
        'accuracy': metrics_report.get('accuracy', 0),
        'macro_f1': metrics_report.get('macro_f1', 0)
    })

if 'fusion_results' in locals():
    tracker.log_metrics({'fusion_accuracy': fusion_results.get('test_accuracy', 0)})

tracker.end_run()
print("✅ Experiment logged to MLflow.")
