# 🧠 Mini-Projet Deep Learning : Conception, Analyse et Optimisation d'un MLP (Fashion-MNIST)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![D2L](https://img.shields.io/badge/Library-d2l--1.0.3-green.svg)](https://d2l.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-orange.svg)](https://github.com/zalandoresearch/fashion-mnist)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Ce dépôt contient le développement complet, l'analyse interne et l'optimisation expérimentale d'un **Perceptron Multicouche (MLP - Multi-Layer Perceptron)** appliqué à la classification d'images sur le jeu de données **Fashion-MNIST**.

L'intégralité du projet (implémentations, vérifications mathématiques, campagnes expérimentales, visualisations et analyse critique) est centralisée dans le notebook unique : **[`MLP.ipynb`](./Projet/MLP.ipynb)** (ou [`MLP.ipynb`](./MLP.ipynb)).

---

## 📋 Table des Matières

- [1. Présentation du Projet & Jeu de Données](#1-présentation-du-projet--jeu-de-données)
- [2. Structure du Dépôt](#2-structure-du-dépôt)
- [3. Architecture et Pipeline du MLP](#3-architecture-et-pipeline-du-mlp)
- [4. Contenu Détaillé du Notebook `MLP.ipynb`](#4-contenu-détaillé-du-notebook-mlpipynb)
  - [Partie I & II : Implémentation From Scratch vs Concise](#partie-i--ii--implémentation-from-scratch-vs-concise)
  - [Partie III & IV : Inspection Interne, Gradients & Autograd](#partie-iii--iv--inspection-interne-gradients--autograd)
  - [Partie V : Étude Expérimentale Approfondie](#partie-v--étude-expérimentale-approfondie)
  - [Partie VI : Discussion Critique & Limites du MLP](#partie-vi--discussion-critique--limites-du-mlp)
  - [Partie VII : Questions de Synthèse Théorique](#partie-vii--questions-de-synthèse-théorique)
- [5. Résultats Expérimentaux Clés](#5-résultats-expérimentaux-clés)
- [6. Configuration Optimale Recommandée](#6-configuration-optimale-recommandée)
- [7. Installation et Exécution](#7-installation-et-exécution)
- [8. Dépendances Principales](#8-dépendances-principales)

---

## 1. Présentation du Projet & Jeu de Données

L'objectif de ce projet est de démystifier le fonctionnement d'un réseau de neurones profond en passant d'une **implémentation bas-niveau mathématique (From Scratch)** à une **implémentation concise modulaire avec PyTorch**, tout en analysant l'impact des hyperparamètres et des techniques modernes de régularisation et d'optimisation.

### Dataset : Fashion-MNIST
- **Nombre d'échantillons :** 60 000 images d'entraînement, 10 000 images de test.
- **Dimensions :** Images en niveaux de gris de taille $28 \times 28$ pixels.
- **Classes (10 catégories d'articles de mode) :** T-shirt/top, Pantalon, Pullover, Robe, Manteau, Sandale, Chemise, Sneaker, Sac, Bottine.
- **Prétraitement :** Aplatissement (*flattening*) de chaque image $28 \times 28$ en un vecteur unidimensionnel de dimension $784$.

---

## 2. Structure du Dépôt

Le projet est conçu pour être entièrement autonome autour du notebook principal :

```text
├── MLP.ipynb               # 📓 Notebook principal contenant tout le code, les analyses et graphiques
├── figures/                # 📊 Graphiques exportés générés lors des campagnes expérimentales
│   ├── MLPScratch.svg
│   ├── MLP.svg
│   ├── Précision (Train vs Test).svg
│   ├── Écart de généralisation (Gap).svg
│   ├── Impact de num_hiddens sur la Précision de Validation.svg
│   ├── Impact du Learning Rate sur la Précision de Validation.svg
│   └── Modèle Baseline vs Configuration Avancée.svg
└── README.md               # 📄 Documentation technique complète du projet
```

---

## 3. Architecture et Pipeline du MLP

Le modèle standard comporte une couche d'entrée, une couche cachée entièrement connectée avec activation non-linéaire ReLU, et une couche de sortie linéaire produisant des scores (*logits*).

```mermaid
flowchart LR
    A["Entrée : Image 28x28 (B, 1, 28, 28)"] --> B["Aplatissement / Flatten (B, 784)"]
    B --> C["Couche Cachée : X · W₁ + b₁ (B, 256)"]
    C --> D["Activation : ReLU = max(0, H) (B, 256)"]
    D --> E["Couche de Sortie : H · W₂ + b₂ (B, 10)"]
    E --> F["Sortie : Logits / Prédictions (B, 10)"]
```

### Dimensions et Paramètres Tenseurs :
- **Entrée vectorisée :** $\mathbf{X} \in \mathbb{R}^{B \times 784}$ (avec un batch size $B=256$)
- **Poids Couche Cachée ($W_1$) :** $(784 \times 256)$ | **Biais ($b_1$) :** $(256)$ $\rightarrow 200\,960$ paramètres
- **Poids Couche de Sortie ($W_2$) :** $(256 \times 10)$ | **Biais ($b_2$) :** $(10)$ $\rightarrow 2\,570$ paramètres
- **Nombre total de paramètres entraînables :** $\mathbf{203\,530}$

---

## 4. Contenu Détaillé du Notebook `MLP.ipynb`

Le notebook [`MLP.ipynb`](./Projet/MLP.ipynb) est structuré en plusieurs parties progressives et autonomes :

### Partie I & II : Implémentation *From Scratch* vs *Concise*

1. **Version *From Scratch* (`MLPScratch`) :**
   - Allocation manuelle des tenseurs de paramètres avec `nn.Parameter` et initialisation gaussienne ($\sigma = 0.01$).
   - Implémentation explicite de la fonction d'activation $\text{ReLU}(x) = \max(0, x)$ avec `torch.max(X, torch.zeros_like(X))`.
   - Remise en forme explicite (`reshape((-1, 784))`) et calcul matriciel explicite (`torch.matmul` / `@`).
2. **Version *Concise* (`MLP`) :**
   - Utilisation des abstractions de haut niveau de PyTorch : `nn.Sequential`, `nn.Flatten()`, `nn.LazyLinear()`, et `nn.ReLU()`.
   - Inférence automatique des dimensions d'entrée grâce aux couches *Lazy*.
3. **Comparaison quantitative :**
   - Les deux versions possèdent exactement **203 530 paramètres entraînables**, confirmant l'équivalence mathématique stricte entre les deux approches.

---

### Partie III & IV : Inspection Interne, Gradients & Autograd

- **Cycle de vie de `nn.LazyLinear` :** Mise en évidence de l'état non-initialisé avant le premier forward pass, puis matérialisation automatique des tenseurs dès l'injection du premier batch.
- **Orientation des matrices :** Validation du stockage PyTorch sous forme `[out_features, in_features]` (ex: $W_1$ stocké en `[256, 784]`).
- **Inspection du mécanisme Autograd :**
  - Calcul de l'erreur via l'entropie croisée (*Cross-Entropy Loss*).
  - Déclenchement de la rétropropagation (`loss.backward()`).
  - Extraction et inspection des gradients (`param.grad`) pour confirmer le calcul effectif des dérivées partielles $\frac{\partial \mathcal{L}}{\partial W}$ et $\frac{\partial \mathcal{L}}{\partial b}$.

---

### Partie V : Étude Expérimentale Approfondie

Cette section constitue le cœur de la validation empirique :

#### 1. Expérience 1 : Variation de la Capacité (`num_hiddens`) & Early Stopping
- **Configurations testées :** $h \in \{16, 32, 64, 128, 256, 512\}$.
- **Observations :**
  - **Sous-apprentissage (Underfitting) :** Pour $h \in \{16, 32\}$, la capacité de représentation est insuffisante (précision plafonnée à ~81–84%).
  - **Zone Optimale :** Pour $h = 256$, le modèle atteint un pic d'environ **86.79%** de précision test.
  - **Surapprentissage (Overfitting) :** Pour $h = 512$, la précision d'entraînement diverge du jeu de test (gap de généralisation accru).
  - **Early Stopping :** Arrêt prématuré paramétré avec une patience de 3 époques, prévenant efficacement la dégradation des performances lors des divergences.

#### 2. Expérience 2 : Sensibilité au Taux d'Apprentissage ($lr$)
- **Configurations testées :** $lr \in \{0.01, 0.1, 0.5\}$.
- **Résultats :**
  - $lr = 0.01$ : Apprentissage beaucoup trop lent, n'atteignant que ~82.08% à la 20ᵉ époque.
  - $lr = 0.1$ : Convergence équilibrée, fluide et robuste (~86.79%).
  - $lr = 0.5$ : Instabilité et sauts erratiques (*overshooting*) dans l'espace des pertes.

#### 3. Expérience 3 : Modèle Baseline vs Configuration Avancée
Comparaison entre :
- **Baseline :** SGD classique, initialisation aléatoire par défaut, aucun mécanisme de régularisation.
- **Configuration Avancée :**
  - Initialisation de **Xavier (Glorot)** pour stabiliser la variance des activations et gradients.
  - **Dropout ($p = 0.3$)** pour limiter la co-adaptation des neurones.
  - **Pénalisation $L_2$ (Weight Decay $= 10^{-4}$)** pour borner la norme des poids.
  - Optimiseur adaptatif **Adam ($lr = 0.001$)**.

---

### Partie VI : Discussion Critique & Limites du MLP

- **Pourquoi le MLP fonctionne-t-il sur Fashion-MNIST ?**
  Les images sont normalisées, centrées et de taille identique. Les corrélations statistiques entre pixels à coordonnées fixes permettent au MLP d'apprendre des motifs globaux malgré la vectorisation (1D).
- **Limites fondamentales du MLP en Vision :**
  1. *Perte de la localité spatiale :* L'opération Flatten supprime les relations topologiques 2D de voisinage entre pixels.
  2. *Explosion combinatoire :* Pour des images haute résolution (ex: $1920 \times 1080$), une couche dense mènerait à plusieurs millions de connexions par neurone.
  3. *Sensibilité aux translations :* Un décalage de quelques pixels modifie entièrement le vecteur d'entrée, nécessitant un réapprentissage complet.
- **Pertinence des CNN :** Utilisation du partage de poids (*weight sharing*) et d'invariance par translation par convolution spatiale.

---

### Partie VII : Questions de Synthèse Théorique

Le notebook répond aux questions conceptuelles clés du Deep Learning :
1. **Nature non-linéaire du MLP :** Rôle des fonctions d'activation pour casser la linéarité et permettre l'approximation universelle de fonctions continues.
2. **Inutilité de couches linéaires sans activation :** Démonstration mathématique que $\mathbf{Y} = (\mathbf{X} W_1) W_2 = \mathbf{X} (W_1 W_2) = \mathbf{X} W_{eq}$ équivaut à une seule transformation linéaire.
3. **Écart de généralisation (Gap) :** Analyse du surapprentissage et justification de l'Early Stopping comme régularisation implicite bornant la norme des poids.
4. **Pédagogie From Scratch vs Conception Industrielle :** Maîtrise des concepts d'algèbre tensorielle versus modularité et optimisations matérielles (CUDA/C++).

---

## 5. Résultats Expérimentaux Clés

| Modèle / Configuration | Optimiseur | Régularisation | Initialisation | Précision Test Max | Comportement & Stabilité |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ($h=256$)** | SGD ($lr=0.1$) | Aucune | Défaut | ~86.0% | Instable, chutes brutales de validation |
| **Faible Capacité ($h=32$)** | SGD ($lr=0.1$) | Early Stopping | Défaut | ~85.9% | Sous-apprentissage modéré |
| **Grande Capacité ($h=512$)** | SGD ($lr=0.1$) | Early Stopping | Défaut | ~86.2% | Surapprentissage visible (gap important) |
| **Modèle Avancé ($h=256$)** | **Adam ($lr=10^{-3}$)** | **Dropout (0.3) + L2 ($10^{-4}$)** | **Xavier** | **> 88.2%** | **Convergence ultra-rapide (84% dès l'ép. 1), courbe parfaitement lisse** |

---

## 6. Configuration Optimale Recommandée

Sur la base de l'ensemble des métriques collectées, la configuration optimale retenue est :

```python
import torch.nn as nn

optimal_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 10)
)
```

- **Optimiseur :** `torch.optim.Adam(optimal_model.parameters(), lr=1e-3, weight_decay=1e-4)`
- **Initialisation :** `nn.init.xavier_uniform_` sur les poids linéaires
- **Arrêt anticipé :** Early Stopping avec patience de 3 époques sur la validation loss/accuracy.

---

## 7. Installation et Exécution

### Prérequis
Assurez-vous de disposer de Python 3.10 ou supérieur installé sur votre machine.

### Installation des bibliothèques
Dans votre terminal ou invite de commande, installez les dépendances requises :

```bash
pip install torch torchvision numpy matplotlib pandas requests d2l==1.0.3
```

### Lancement du Notebook
Lancez votre environnement Jupyter préféré :

```bash
jupyter notebook MLP.ipynb
```
*ou avec JupyterLab / VS Code : ouvrez directement [`MLP.ipynb`](./Projet/MLP.ipynb).*

Vous pouvez exécuter les cellules séquentiellement (`Run All`) pour reproduire l'intégralité des entraînements, des courbes comparatives et des métriques.

---

## 8. Dépendances Principales

- [PyTorch](https://pytorch.org/) (Tenseurs, `torch.nn`, autograd, optimiseurs)
- [d2l (Dive into Deep Learning)](https://d2l.ai/) (DataLoaders Fashion-MNIST, abstractions d'entraînement)
- [Matplotlib](https://matplotlib.org/) (Génération et exportation des courbes SVG)
- [NumPy](https://numpy.org/) (Calcul matriciel et manipulations vectorielles)

---
*Projet réalisé dans le cadre du module Deep Learning (Master / Cycle Ingénieur - 5DSIG2 - Semestre 9).*

