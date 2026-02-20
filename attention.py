# ==============================================
# FICHIER: app_attention.py
# ==============================================
# Application professionnelle - CNN avec Attention
# Pour exécuter: streamlit run app_attention.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import time
import pandas as pd
from datetime import datetime
import hashlib

# ==============================================
# CONFIGURATION DE LA PAGE
# ==============================================

st.set_page_config(
    page_title="Flowerspredict",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================
# CSS PERSONNALISÉ - DESIGN PREMIUM
# ==============================================

st.markdown("""
<style>
    /* Importation des polices */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Variables de couleur - Thème élégant */
    :root {
        --primary: #8B5CF6;
        --primary-light: #A78BFA;
        --primary-dark: #7C3AED;
        --secondary: #EC4899;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --dark: #1F2937;
        --light: #F9FAFB;
        --gray: #6B7280;
    }
    
    /* Style global */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Conteneur principal */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* En-tête avec gradient */
    .premium-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .premium-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
    }
    
    .premium-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        position: relative;
    }
    
    /* Badge du modèle */
    .model-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    /* Cartes d'information */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.2);
        border-color: var(--primary-light);
    }
    
    .info-card h3 {
        color: var(--primary);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Zone de résultat */
    .result-area {
        background: linear-gradient(135deg, #F3E8FF 0%, #FFE6F0 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
    }
    
    .prediction-box {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(139, 92, 246, 0.2);
    }
    
    .prediction-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--gray);
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--success);
    }
    
    /* Barre de progression personnalisée */
    .progress-container {
        width: 100%;
        height: 10px;
        background: #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Métriques */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Bouton personnalisé */
    .custom-button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 30px -10px rgba(139, 92, 246, 0.5);
    }
    
    .custom-button::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .custom-button:hover::after {
        opacity: 1;
    }
    
    /* Footer */
    .premium-footer {
        text-align: center;
        padding: 2rem;
        color: white;
        font-size: 0.9rem;
        margin-top: 3rem;
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-slide {
        animation: slideIn 0.6s ease-out;
    }
    
    /* Tooltip moderne */
    .tooltip-modern {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip-modern .tooltip-text {
        visibility: hidden;
        background: var(--dark);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        font-size: 0.8rem;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
    }
    
    .tooltip-modern:hover .tooltip-text {
        visibility: visible;
    }
    
    /* File uploader personnalisé */
    .uploadfile {
        border: 2px dashed var(--primary-light);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(139, 92, 246, 0.05);
        transition: all 0.3s;
    }
    
    .uploadfile:hover {
        border-color: var(--primary);
        background: rgba(139, 92, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# DÉFINITION DU MODÈLE AVEC ATTENTION
# ==============================================

class ChannelAttention(nn.Module):
    """Attention channel-wise (Squeeze-and-Excitation)"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    """Attention spatiale"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

class CBAMBlock(nn.Module):
    """Bloc d'attention complet"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionCNN(nn.Module):
    """CNN avec mécanismes d'attention - CBAM"""
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
        
        # Bloc 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attention1 = CBAMBlock(32)
        
        # Bloc 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention2 = CBAMBlock(64)
        
        # Bloc 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attention3 = CBAMBlock(128)
        
        # Bloc 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attention4 = CBAMBlock(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.attention3(x)
        
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ==============================================
# FONCTIONS DE VISUALISATION AVEC PLOTLY
# ==============================================

def create_gauge_chart(confidence, title="Niveau de confiance"):
    """Crée une jauge de confiance interactive"""
    
    colors = ['#EF4444', '#F59E0B', '#10B981']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'family': 'Plus Jakarta Sans'}},
        delta={'reference': 50, 'position': "top"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "rgba(139, 92, 246, 0.8)"},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 33], 'color': colors[0]},
                {'range': [33, 66], 'color': colors[1]},
                {'range': [66, 100], 'color': colors[2]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Plus Jakarta Sans'}
    )
    
    return fig

def create_probability_bars(probs, class_names, predicted_class):
    """Crée un graphique à barres des probabilités"""
    
    colors = ['#8B5CF6' if cls == predicted_class else '#D1D5DB' 
              for cls in class_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=class_names,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs],
            textposition='outside',
            textfont=dict(size=12, family='Plus Jakarta Sans'),
            hovertemplate='<b>%{y}</b><br>Probabilité: %{x:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Distribution des probabilités",
            'font': {'size': 18, 'family': 'Plus Jakarta Sans', 'color': '#1F2937'}
        },
        xaxis={
            'title': "Probabilité",
            'range': [0, 1],
            'tickformat': '.0%',
            'gridcolor': '#E5E7EB',
            'zerolinecolor': '#E5E7EB'
        },
        yaxis={
            'title': None,
            'gridcolor': '#E5E7EB'
        },
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Plus Jakarta Sans'}
    )
    
    return fig

def create_attention_heatmap(feature_maps):
    """Crée une heatmap des features maps"""
    
    if feature_maps is None:
        return None
    
    # Moyenne sur les canaux pour visualisation
    attention_map = feature_maps.mean(dim=1).squeeze().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=attention_map,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='Intensité: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Carte d'attention",
            'font': {'size': 16, 'family': 'Plus Jakarta Sans'}
        },
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

# ==============================================
# CHARGEMENT DU MODÈLE
# ==============================================

@st.cache_resource(show_spinner=False)
def load_attention_model():
    """Charge le modèle d'attention avec animation"""
    
    model_path = "sn_attention.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle '{model_path}' non trouvé!")
        return None, None, None
    
    try:
        # Barre de progression stylisée
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("📦 **Chargement des poids du modèle...**")
        progress_bar.progress(30)
        
        # Chargement du checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        progress_bar.progress(60)
        
        # Récupération des métadonnées
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            num_classes = len(class_names)
        else:
            class_names = [f"Classe_{i}" for i in range(3)]
            num_classes = 3
        
        status_text.markdown("🔧 **Initialisation de l'architecture...**")
        progress_bar.progress(80)
        
        # Création et chargement du modèle
        model = AttentionCNN(num_classes)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        progress_bar.progress(100)
        
        status_text.markdown("✅ **Modèle chargé avec succès!**")
        time.sleep(0.5)
        
        # Nettoyage
        progress_bar.empty()
        status_text.empty()
        
        return model, class_names, num_classes
    
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        return None, None, None

# ==============================================
# FONCTION DE PRÉDICTION
# ==============================================

def predict_with_attention(model, image, class_names):
    """Prédiction avec capture des cartes d'attention"""
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Prétraitement
    input_tensor = transform(image).unsqueeze(0)
    
    # Capture des activations d'attention
    attention_maps = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            attention_maps[name] = output.detach().cpu()
        return hook
    
    # Enregistrement des hooks
    handles = []
    for i, attention_block in enumerate([model.attention1, model.attention2, 
                                         model.attention3, model.attention4]):
        handle = attention_block.register_forward_hook(hook_fn(f'attention_{i+1}'))
        handles.append(handle)
    
    # Prédiction
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    processing_time = time.time() - start_time
    
    # Nettoyage des hooks
    for handle in handles:
        handle.remove()
    
    return {
        'class': class_names[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': probabilities[0].numpy(),
        'attention_maps': attention_maps.get('attention_4', None),  # Dernière couche
        'processing_time': processing_time
    }

# ==============================================
# INTERFACE PRINCIPALE
# ==============================================

def main():
    # Conteneur principal
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # En-tête premium
    st.markdown("""
    <div class="premium-header animate-slide">
        <h1>🎯 Flowers Predict Pro</h1>
        <p>Classification intelligente avec mécanismes d'attention CBAM</p>
        <div class="model-badge">
            <span>🤖  Architecture: CBAM </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement du modèle
    with st.spinner("🚀 Initialisation du modèle..."):
        model, class_names, num_classes = load_attention_model()
    
    if model is None:
        st.markdown("""
        <div style="background: #FEE2E2; padding: 2rem; border-radius: 20px; text-align: center;">
            <h3 style="color: #DC2626;">❌ Modèle non trouvé</h3>
            <p style="color: #6B7280;">Veuillez placer le fichier <code>sn_attention.pth</code> dans le dossier de l'application</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Liste des fichiers .pth disponibles
        st.markdown("### 📂 Fichiers trouvés:")
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            for f in pth_files:
                st.markdown(f"- `{f}`")
        else:
            st.markdown("*Aucun fichier .pth trouvé*")
        
        st.stop()
    
    # Layout principal
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        # Zone d'upload premium
        st.markdown("### 📤 Import d'image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Formats supportés: JPG, PNG, BMP, TIFF"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Métriques de l'image
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Largeur</div>
                </div>
                """.format(image.size[0]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Hauteur</div>
                </div>
                """.format(image.size[1]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Format</div>
                </div>
                """.format(uploaded_file.type.split('/')[-1].upper()), unsafe_allow_html=True)
            
            # Affichage de l'image
            st.image(image, caption="Image sélectionnée", use_container_width=True)
            
            # Bouton d'analyse
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 ANALYSER ", key="analyze", use_container_width=True):
                with st.spinner("🔄 Calcul des cartes d'attention..."):
                    result = predict_with_attention(model, image, class_names)
                    
                    # Stockage dans session state
                    st.session_state['last_prediction'] = result
                    st.session_state['last_image'] = image
                    st.session_state['prediction_time'] = datetime.now().strftime("%H:%M:%S")
                    
                    st.rerun()
    
    with col_right:
        # Informations sur le modèle
        st.markdown("### 🧠 Architecture du modèle")
        
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Mécanismes d'attention</h3>
            <div style="margin: 1rem 0;">
                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                    <span style="background: #8B5CF6; color: white; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.8rem;">Channel Attention</span>
                    <span style="background: #EC4899; color: white; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.8rem;">Spatial Attention</span>
                    <span style="background: #10B981; color: white; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.8rem;">CBAM Blocks</span>
                </div>
            </div>
            <p><strong>Nombre de classes:</strong> {}</p>
            <p><strong>Classes:</strong> {}</p>
        </div>
        """.format(num_classes, ', '.join(class_names)), unsafe_allow_html=True)
        
        # Guide rapide
        st.markdown("### 📖 Guide d'utilisation")
        
        st.markdown("""
        <div class="info-card">
            <h3>Étapes</h3>
            <ol style="color: #4B5563;">
                <li>Téléchargez une image de coton</li>
                <li>Cliquez sur "Analyser avec attention"</li>
                <li>Visualisez les cartes d'attention</li>
                <li>Interprétez les résultats</li>
            </ol>
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.9rem;">
                    <span style="color: #8B5CF6;">💡 Les zones rouges</span> indiquent où le modèle concentre son attention
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Affichage des résultats
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Résultats de l'analyse")
        
        result = st.session_state['last_prediction']
        image = st.session_state['last_image']
        
        # Zone de résultat premium
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            # Image avec overlay d'attention si disponible
            st.image(image, caption="Image analysée", use_container_width=True)
            
            # Métriques
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{:.2f}s</div>
                    <div class="metric-label">Temps</div>
                </div>
                """.format(result['processing_time']), unsafe_allow_html=True)
            
            with col_m2:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Classe</div>
                </div>
                """.format(result['class'][:10] + "..." if len(result['class']) > 10 else result['class']), unsafe_allow_html=True)
            
            with col_m3:
                st.markdown("""
                <div class="metric-item">
                    <div class="metric-value">{:.1%}</div>
                    <div class="metric-label">Confiance</div>
                </div>
                """.format(result['confidence']), unsafe_allow_html=True)
        
        with col_res2:
            # Boîte de prédiction
            confidence_class = "high" if result['confidence'] > 0.7 else "medium" if result['confidence'] > 0.4 else "low"
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Résultat de la classification</div>
                <div class="prediction-value">{result['class']}</div>
                <div class="confidence-value">{result['confidence']:.1%}</div>
                <div style="margin: 1rem 0;">
                    <span class="tooltip-modern">
                        ℹ️
                        <span class="tooltip-text">Niveau de confiance du modèle</span>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphiques
        st.markdown("### 📈 Analyse détaillée")
        
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            # Jauge de confiance
            fig_gauge = create_gauge_chart(result['confidence'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_graph2:
            # Barres de probabilités
            fig_bars = create_probability_bars(
                result['probabilities'], 
                class_names, 
                result['class']
            )
            st.plotly_chart(fig_bars, use_container_width=True)
        
        # Carte d'attention
        if result['attention_maps'] is not None:
            st.markdown("### 🔍 Visualisation de l'attention")
            
            fig_attention = create_attention_heatmap(result['attention_maps'])
            if fig_attention:
                st.plotly_chart(fig_attention, use_container_width=True)
                
                st.markdown("""
                <div style="background: #F3F4F6; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <p style="margin: 0; color: #4B5563;">
                        <strong>🔬 Interprétation:</strong> Les zones les plus claires (jaunes) montrent où le modèle concentre son attention pour prendre sa décision.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Statistiques d'utilisation
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Statistiques session")
        
        # Simuler quelques statistiques
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.markdown("""
            <div class="metric-item">
                <div class="metric-value">1</div>
                <div class="metric-label">Prédictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown("""
            <div class="metric-item">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Conf. moyenne</div>
            </div>
            """.format(result['confidence']), unsafe_allow_html=True)
        
        with col_stat3:
            st.markdown("""
            <div class="metric-item">
                <div class="metric-value">{}</div>
                <div class="metric-label">Classes</div>
            </div>
            """.format(num_classes), unsafe_allow_html=True)
        
        with col_stat4:
            st.markdown("""
            <div class="metric-item">
                <div class="metric-value">{}</div>
                <div class="metric-label">Dernière</div>
            </div>
            """.format(st.session_state.get('prediction_time', '-')), unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="premium-footer">
        <p>AttentionCNN Pro - Classification intelligente avec mécanismes d'attention CBAM</p>
        <p style="font-size: 0.8rem; opacity: 0.8;">Développé avec PyTorch • Streamlit • Plotly</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fermeture du conteneur
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================
# LANCEMENT
# ==============================================

if __name__ == "__main__":

    main()
