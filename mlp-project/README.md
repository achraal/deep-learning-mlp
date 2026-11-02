# 🧠 Mini-Projet Deep Learning : Conception, Analyse et Optimisation d'un MLP (Fashion-MNIST)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![D2L](https://img.shields.io/badge/Library-d2l--1.0.3-green.svg)](https://d2l.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-orange.svg)](https://github.com/zalandoresearch/fashion-mnist)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Ce répertoire contient l'implémentation, l'analyse interne et l'optimisation expérimentale d'un **Perceptron Multicouche (MLP - Multi-Layer Perceptron)** pour la classification d'images sur le jeu de données **Fashion-MNIST**.

L'intégralité du projet est implémentée et documentée dans le notebook unique : **[`MLP.ipynb`](./MLP.ipynb)**.

---

## 📋 Table des Matières

- [1. Présentation du Projet & Jeu de Données](#1-présentation-du-projet--jeu-de-données)
- [2. Structure du Projet](#2-structure-du-projet)
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

L'objectif de ce projet est d'explorer en profondeur la mécanique des réseaux de neurones denses :
- D'une **implémentation bas-niveau mathématique (From Scratch)** sans modules préconçus,
- À une **implémentation concise et modulaire avec PyTorch**,
- En passant par l'analyse fine des dimensions de tenseurs, du graphe de calcul computationnel `autograd` et de campagnes expérimentales d'optimisation.

### Dataset : Fashion-MNIST
- **Échantillons :** 60 000 images d'entraînement, 10 000 images de test.
- **Résolution :** Images $28 \times 28$ pixels en niveaux de gris (1 canal).
- **Classes (10 catégories de vêtements) :** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- **Prétraitement :** Aplatissement (*Flattening*) en un vecteur 1D de taille $784$.

---

## 2. Structure du Projet

```text
├── MLP.ipynb               # 📓 Notebook principal (code complet, entraînement, visualisations et analyse)
├── figures/                # 📊 Graphiques générés lors des campagnes expérimentales
│   ├── MLPScratch.svg
│   ├── MLP.svg
│   ├── Précision (Train vs Test).svg
│   ├── Écart de généralisation (Gap).svg
│   ├── Impact de num_hiddens sur la Précision de Validation.svg
│   ├── Impact du Learning Rate sur la Précision de Validation.svg
│   └── Modèle Baseline vs Configuration Avancée.svg
└── README.md               # 📄 Documentation technique complète
```

---

## 3. Architecture et Pipeline du MLP

```mermaid
flowchart LR
    A["Entrée : Image 28x28 (B, 1, 28, 28)"] --> B["Aplatissement / Flatten (B, 784)"]
    B --> C["Couche Cachée : X · W₁ + b₁ (B, 256)"]
    C --> D["Activation : ReLU = max(0, H) (B, 256)"]
    D --> E["Couche de Sortie : H · W₂ + b₂ (B, 10)"]
    E --> F["Sortie : Logits / Prédictions (B, 10)"]
```

### Dimensions et Paramètres Tenseurs :
- **Entrée vectorisée :** $\mathbf{X} \in \mathbb{R}^{B \times 784}$ (taille de batch standard $B=256$)
- **Paramètres Couche Cachée :** Poids $W_1 \in \mathbb{R}^{784 \times 256}$, Biais $b_1 \in \mathbb{R}^{256}$ $\rightarrow 200\,960$ paramètres
- **Paramètres Couche de Sortie :** Poids $W_2 \in \mathbb{R}^{256 \times 10}$, Biais $b_2 \in \mathbb{R}^{10}$ $\rightarrow 2\,570$ paramètres
- **Total de paramètres entraînables :** $\mathbf{203\,530}$

---

## 4. Contenu Détaillé du Notebook `MLP.ipynb`

### Partie I & II : Implémentation *From Scratch* vs *Concise*

1. **Version *From Scratch* (`MLPScratch`) :**
   - Allocation manuelle des tenseurs de poids et de biais avec `nn.Parameter` et écart-type $\sigma = 0.01$.
   - Implémentation explicite de l'activation non-linéaire $\text{ReLU}(X) = \max(0, X)$ via `torch.max(X, torch.zeros_like(X))`.
   - Remise en forme manuelle (`reshape((-1, 784))`) et multiplication matricielle directe (`@` / `matmul`).
2. **Version *Concise* (`MLP`) :**
   - Implémentation via l'écosystème PyTorch haut-niveau : `nn.Sequential`, `nn.Flatten()`, `nn.LazyLinear()`, et `nn.ReLU()`.
   - Inférence automatique des dimensions grâce au paradigme *lazy*.
3. **Comparaison quantitative :**
   - Validation de la concordance stricte : exactement **203 530 paramètres entraînables** dans les deux cas.

---

### Partie III & IV : Inspection Interne, Gradients & Autograd

- **Comportement de `nn.LazyLinear` :** Démonstration de l'état non-initialisé initial puis de la matérialisation immédiate après le passage du premier batch (`forward pass`).
- **Convention de stockage matriciel :** Vérification de la convention PyTorch `[out_features, in_features]`.
- **Validation du graphe de calcul (`Autograd`) :**
  - Calcul de l'erreur via l'entropie croisée (*Cross-Entropy Loss*).
  - Déclenchement de la rétropropagation (`loss.backward()`).
  - Vérification de la présence et de la non-nullité des gradients `param.grad` prêts pour la descente de gradient.

---

### Partie V : Étude Expérimentale Approfondie

#### 1. Expérience 1 : Variation de la Capacité (`num_hiddens`) & Early Stopping
- **Plage testée :** $h \in \{16, 32, 64, 128, 256, 512\}$.
- **Observations :**
  - **Sous-apprentissage :** Pour $h \in \{16, 32\}$, le modèle manque de paramètres expressifs (précision ~81-84%).
  - **Capacité optimale :** Pour $h = 256$, la précision test culmine à **86.79%**.
  - **Surapprentissage :** Pour $h = 512$, un écart de généralisation (*gap*) marqué apparaît entre train et validation.
  - **Early Stopping :** Arrêt automatique (patience = 3 époques), protégeant le modèle contre la sur-mémorisation.

#### 2. Expérience 2 : Sensibilité au Taux d'Apprentissage ($lr$)
- **Plage testée :** $lr \in \{0.01, 0.1, 0.5\}$.
- **Observations :**
  - $lr = 0.01$ : Convergence excessivement lente (~82.08% à l'époque 20).
  - $lr = 0.1$ : Taux optimal assurant une descente fluide et efficace (~86.79%).
  - $lr = 0.5$ : Instabilités numériques et fluctuations oscillatoires d'une époque à l'autre (*overshooting*).

#### 3. Expérience 3 : Modèle Baseline vs Configuration Avancée
- **Modèle Baseline :** SGD classique, initialisation aléatoire standard, absence de régularisation.
- **Configuration Avancée :**
  - **Initialisation Xavier / Glorot :** Maintien d'une variance stable des gradients dès la couche d'entrée.
  - **Dropout ($p = 0.3$) :** Désactivation stochastique de neurones pour briser les co-adaptations néfastes.
  - **Pénalisation $L_2$ (Weight Decay $= 10^{-4}$) :** Contrainte de régularisation lissant la dynamique des poids.
  - **Optimiseur Adam ($lr = 0.001$) :** Pas d'apprentissage adaptatif par paramètre via estimation des moments.

---

### Partie VI : Discussion Critique & Limites du MLP

- **Pourquoi le MLP fonctionne-t-il sur Fashion-MNIST ?**
  Fashion-MNIST présente des images cadrées, centrées et uniformes. Les corrélations statistiques entre pixels à indices fixes suffisent pour la classification globale.
- **Limites du MLP en Vision par Ordinateur :**
  1. *Perte de la localité 2D :* Aucune préservation des relations spatiales de voisinage (bords, textures).
  2. *Explosion paramétrique :* Inapplicable aux images de haute dimension sous peine de saturer la mémoire et de surapprendre massivement.
  3. *Manque d'invariance par translation :* Une translation de quelques pixels invalide les motifs appris par les neurones denses.
- **Apport des architectures convolutives (CNN) :** Utilisation de noyaux convolutifs partagés préservant la topologie spatiale et conférant l'invariance par translation.

---

### Partie VII : Questions de Synthèse Théorique

1. **Non-linéarité :** L'activation non-linéaire permet au réseau de briser la linéarité affine et d'approximer n'importe quelle variété continue (théorème d'approximation universelle).
2. **Succession de couches linéaires :** Sans activation intermédiaire, $\mathbf{Y} = \mathbf{X} W_1 W_2 = \mathbf{X} W_{eq}$, le réseau équivaut strictement à un simple modèle linéaire sans gain d'expressivité.
3. **Early Stopping & Régularisation :** Forme implicite de contrainte $L_2$ empêchant les poids d'atteindre des amplitudes extrêmes au cours des époques tardives.

---

## 5. Résultats Expérimentaux Clés

| Modèle / Configuration | Optimiseur | Régularisation | Initialisation | Précision Test Max | Comportement & Stabilité |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline ($h=256$)** | SGD ($lr=0.1$) | Aucune | Défaut | ~86.0% | Instable, chutes brutales de validation |
| **Faible Capacité ($h=32$)** | SGD ($lr=0.1$) | Early Stopping | Défaut | ~85.9% | Sous-apprentissage modéré |
| **Grande Capacité ($h=512$)** | SGD ($lr=0.1$) | Early Stopping | Défaut | ~86.2% | Surapprentissage visible (gap important) |
| **Modèle Avancé ($h=256$)** | **Adam ($lr=10^{-3}$)** | **Dropout (0.3) + L2 ($10^{-4}$)** | **Xavier** | **> 88.2%** | **Convergence fulgurante (84% dès l'ép. 1), courbe parfaitement lisse** |

---

## 6. Configuration Optimale Recommandée

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
- **Initialisation :** `nn.init.xavier_uniform_` sur les couches denses
- **Garde-fou :** Early Stopping avec patience de 3 époques.

---

## 7. Installation et Exécution

### Installation des dépendances
```bash
pip install torch torchvision numpy matplotlib pandas requests d2l==1.0.3
```

### Lancement du Notebook
```bash
jupyter notebook MLP.ipynb
```

Toutes les cellules s'exécutent de façon linéaire et autonome pour reproduire les entraînements et générer les graphiques d'analyse.

---

## 8. Dépendances Principales

- **PyTorch** (`torch`, `torch.nn`)
- **D2L** (`d2l==1.0.3`)
- **NumPy**
- **Matplotlib**

---
*Projet Deep Learning (5DSIG2 - S9).*

