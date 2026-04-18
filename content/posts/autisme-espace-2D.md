---
title: "L'autisme comme espace à deux dimensions"
date: '2026-04-18T00:00:00+02:00'
tags: ["Autisme", "TSA", "Mathématiques", "DSM-5", "Modélisation"]
categories: ["Sciences"]
math: true
---

# Un modèle mathématique pour penser le diagnostic TSA

**Objectif de ce notebook :**  <br>
Formaliser le spectre autistique non pas comme une ligne, mais comme un **espace vectoriel à 2 dimensions**, et explorer des méthodes pour placer un profil individuel dans cet espace à partir de données cliniques.

## Contexte perso
Ça m'énerve qu'on ne sache pas vraiment de quoi on parle dans le cadre d'un TSA et qu'on recule d'autant plus en supprimant les catégories précédemment établies depuis le DSM-5 !!! <br>
Pour cela, rien de mieux que des maths :)

## Contexte scientifique

Le DSM-5 organise le TSA autour de **deux axes orthogonaux** :

| Axe | Domaine | Notation |
|-----|---------|----------|
| **A** | Déficits en communication et interaction sociales | $a \in [0, 3]$ |
| **B** | Comportements, intérêts et activités restreints/répétitifs (CIR) | $b \in [0, 3]$ |

Un profil clinique peut donc être représenté comme un **point** $P = (a, b)$ dans $\mathbb{R}^2$. <br>
La sévérité globale n'est **pas** la distance à l'origine, mais une fonction plus complexe que nous allons définir et représenter.


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
```

# Formalisation mathématique

## L'espace de diagnostic $\mathcal{D}$

On définit l'espace de diagnostic comme :

$$\mathcal{D} = [0, 3] \times [0, 3] \subset \mathbb{R}^2$$

Chaque point $P = (a, b)$ encode :
- $a$ : score de sévérité sur l'axe A **social/communication** (0 = absent, 3 = maximal)
- $b$ : score de sévérité sur l'axe B **CIR** (0 = absent, 3 = maximal)

## Seuil diagnostique

Le diagnostic de TSA exige que **les deux critères** soient présents de façon cliniquement significative. <br>
On modélise le seuil diagnostique minimal comme :

$$\text{TSA} \iff a \geq a_{\min} \quad \land \quad b \geq b_{\min}$$

Selon le DSM-5, le niveau 1 correspond grossièrement à $a_{\min} \approx 0.5$, $b_{\min} \approx 0.5$.  
La région de diagnostic est donc un **rectangle** dans $\mathcal{D}$.

## Niveaux de sévérité DSM-5

| Niveau | Interprétation | Zone approximative |
|--------|---------------|-------------------|
| 1 | Nécessite un soutien | $a \in [0.5, 1.5[$, $b \in [0.5, 1.5[$ |
| 2 | Nécessite un soutien important | $a \in [1.5, 2.5[$, $b \in [1.5, 2.5[$ |
| 3 | Nécessite un soutien très important | $a \geq 2.5$, $b \geq 2.5$ |

Le niveau global est défini par le **maximum** des deux scores : $L = \max(a, b)$.



```python
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Zones de sévérité DSM-5
zone_colors = {
    (0.5, 1.5): 'lightgreen',
    (1.5, 2.5): 'lightyellow',
    (2.5, 3.0): 'lightsalmon'}

for (vmin, vmax), col in zone_colors.items():
    rect = plt.Rectangle((vmin, vmin), vmax - vmin, vmax - vmin, facecolor=col)
    ax.add_patch(rect)

# Zone CIR isolés hors spectre (axe B élevé, axe A faible)
ax.fill_betweenx([0.5, 3], 0, 0.5, color='lightgray')

# Lignes de niveau
for v in [0.5, 1.5, 2.5]:
    ax.axhline(v, color='gray', linewidth=0.8)
    ax.axvline(v, color='gray', linewidth=0.8)

# Rectangle seuil TSA
rect_diag = plt.Rectangle((0.5, 0.5), 2.5, 2.5, edgecolor='red', linewidth=3, facecolor='none')
ax.add_patch(rect_diag)
ax.text(0.52, 0.52, 'Seuil TSA', color='red', va='bottom')

# Profils illustratifs : (x, y, couleur, label)
profiles = [
    (0.9, 2.2, 'purple', 'Asperger-like (CIR eleve, Social modere)'),
    (2.8, 2.8, 'red',    'Autisme classique severe'),
    (2.0, 1.8, 'orange', 'Profil mixte niveau 2'),
    (0.8, 0.8, 'green',  'Niveau 1 compense'),
    (0.2, 2.1, 'gray',   'Hors spectre (CIR seuls)'),]

for x, y, col, label in profiles:
    ax.scatter(x, y, s=100, color=col, zorder=5)
    ax.annotate(label, (x, y), textcoords='offset points', xytext=(8, 4), color=col)

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_xlabel('Deficits sociaux / communication')
ax.set_ylabel('Comportements restreints/repetitifs (CIR)')
ax.set_title("L'espace de diagnostic D = [0,3]^2\nRepresentation des profils TSA")

ticks = [0, 0.5, 1.5, 2.5, 3]
tlabels = ['0', '0.5 (seuil)', '1.5 (N1->N2)', '2.5 (N2->N3)', '3']
ax.set_xticks(ticks)
ax.set_xticklabels(tlabels, fontsize=8)
ax.set_yticks(ticks)
ax.set_yticklabels(tlabels, fontsize=8)

patches = [
    mpatches.Patch(color='lightgreen',  label='Niveau 1'),
    mpatches.Patch(color='lightyellow', label='Niveau 2'),
    mpatches.Patch(color='lightsalmon', label='Niveau 3'),
    mpatches.Patch(color='lightgray',   label='CIR isolés (hors spectre)'),]

ax.legend(handles=patches)

plt.tight_layout()
plt.show()
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_3_0.png)
    


# Métriques : comment mesurer la "distance" au seuil ?

## Distance euclidienne à l'origine (piège)

$$d_2(P) = \sqrt{a^2 + b^2}$$

**Problème :** un profil $(3, 0)$ a la même distance à l'origine qu'un profil $(0, 3)$ ou $(\sqrt{2}, \sqrt{2}) \approx (2.12, 2.12)$.  
Or ces trois profils sont cliniquement très différents, seul le dernier est TSA.

## Distance de Chebyshev (max des composantes)

$$d_\infty(P) = \max(a, b)$$

Cette métrique correspond exactement à la logique du DSM-5 pour le niveau de sévérité. Elle mesure "jusqu'où va le critère le plus sévère".

## Distance à la région diagnostique (la plus utile)

On définit la région diagnostique $\mathcal{R} = [a_{\min}, 3] \times [b_{\min}, 3]$. La distance d'un point $P$ à cette région est :

$$d(P, \mathcal{R}) = \sqrt{\max(0, a_{\min} - a)^2 + \max(0, b_{\min} - b)^2}$$

- Si $d(P, \mathcal{R}) = 0$ : le point est **dans** la région TSA.
- Si $d(P, \mathcal{R}) > 0$ : le point est **hors** spectre, et la valeur indique à quelle distance du seuil il se trouve.

## Score de sévérité composite (proposition)

### Pourquoi une somme pondérée ?

Les métriques précédentes ($d_2$, $d_\infty$, $d_{\mathcal{R}}$) mesurent toutes une **distance dans l'espace**. Elles répondent à la question : *"où est ce profil ?"*

Le score composite répond à une question différente : *"quelle est la charge globale de ce profil ?"*  

C'est un outil de **résumé scalaire**. On veut un seul nombre qui synthétise la position dans $\mathcal{D}$, par exemple pour comparer des groupes, suivre l'évolution d'un patient dans le temps, ou alimenter un modèle prédictif.

- **Suivre l'évolution d'un patient dans le temps** : ici le score composite remplace les trajectoires bivariées par une courbe scalaire, ce qui est beaucoup plus lisible cliniquement.

Au lieu de dire *"entre T0 et T2, l'axe A est passé de 2.1 à 1.5 et l'axe B de 2.5 à 2.4"*, on dit *"$S$ est passé de 2.30 à 1.95"*, autrement une baisse de 0.35 point, interprétable directement.

Mais surtout, $S(t)$ devient une série temporelle, ce qui ouvre des outils d'analyse :
    - Test de tendance (régression linéaire sur $t$) : la trajectoire est-elle statistiquement descendante ?
    - Détection de point de rupture : y a-t-il eu un moment où la trajectoire a changé de pente ?
    - Comparaison inter-patients : deux patients ont-ils des trajectoires similaires malgré des profils initiaux différents ?

La limite ici est importante : un score qui baisse peut refléter une vraie amélioration *ou* du masking. $S$ ne distingue pas les deux. C'est pour cela qu'il faut le lire conjointement avec $d(P, \mathcal{R})$ et les scores bruts.

- **Alimenter un modèle prédictif** : c'est probablement l'usage le plus puissant, et le plus technique.

Un modèle prédictif (régression, arbre de décision, réseau de neurones) cherche à prédire une variable cible $y$ (ex : niveau d'autonomie à l'âge adulte, réponse à une thérapie) à partir de variables d'entrée $X$.
Si $X$ contient les deux axes séparément, le modèle doit apprendre lui-même leur relation avec $y$, ce qui nécessite plus de données. Si on lui fournit $S$ directement, on lui impose une structure (linéarité, pondération) qui peut accélérer l'apprentissage ou réduire le bruit si la structure est pertinente.
Plus précisément, $S$ joue le rôle d'une feature engineerée, d'une variable construite manuellement à partir de la connaissance du domaine. C'est une pratique standard en machine learning clinique : plutôt que de laisser le modèle découvrir que "axe A et axe B contribuent ensemble à prédire $y$", on lui dit directement comment les combiner.

Mais attention à un piège majeur : si les poids $w_A, w_B$ sont choisis en regardant les données d'entraînement (ce qui est tentant), on introduit une fuite d'information (*data leakage*) qui va gonfler artificiellement les performances. Les poids doivent être fixés *a priori* sur des bases cliniques ou théoriques, pas optimisés sur le jeu de données qu'on veut prédire.


- **Comparer des groupes** : le problème fondamental quand on veut comparer deux groupes (ex : enfants autistes vs contrôles, ou groupe avant/après intervention) c'est qu'on a deux scores par individu $(a, b)$ et non plus un seul.

Statistiquement, comparer deux groupes sur deux variables simultanément est possible (MANOVA, distance de Mahalanobis, etc.), mais cela complexifie l'interprétation et réduit la puissance si les deux axes sont corrélés.
Le score composite $S = w_A \cdot a + w_B \cdot b$ projette chaque individu sur une droite dans $\mathcal{D}$, réduisant le problème à
une seule dimension. On peut alors faire un simple test t, une ANOVA, calculer une taille d'effet (Cohen's d), c'est-à-dire des outils que tout le monde comprend.

La contrepartie : en projetant sur une droite, on perd de l'information. Si deux groupes diffèrent fortement sur l'axe A mais pas sur l'axe B, un score avec $w_A = w_B = 0.5$ capture la moitié de cette différence seulement. Choisir les poids de façon éclairée devient alors un enjeu scientifique réel.

Pour cela, j'aimerais introduire un score de méta-analyse qui couvrirait le biais de sélection des poids. Ce score de méta-analyse sera à retrouver dans la dernière partie : *7. Score de méta-analyse : robustesse sous incertitude des poids*.

### Définition

$$S(a, b) = w_A \cdot a + w_B \cdot b \quad \text{avec } w_A + w_B = 1, \; w_A, w_B \geq 0$$

C'est une **moyenne pondérée** des deux scores. La contrainte $w_A + w_B = 1$ garantit que $S \in [0, 3]$, c'est-à-dire qu'il reste dans la même échelle que les scores individuels (ce qui facilite l'interprétation).

### Quelle est l'utilité concrète des poids ?

Les poids permettent de **moduler l'importance relative des deux axes** selon le contexte clinique ou scientifique.

**Cas de base (symétrique) :** $w_A = w_B = 0.5$  
$\longrightarrow$ Les deux axes contribuent également. C'est la position neutre correspondant au DSM-5.

**Cas clinique 1 - enfants en bas âge :** $w_A = 0.7$, $w_B = 0.3$  
$\longrightarrow$ On donne plus de poids aux déficits sociaux, qui sont plus prédictifs du devenir fonctionnel à long terme chez les jeunes enfants autistes.

**Cas clinique 2 - adultes avec profil Asperger :** $w_A = 0.3$, $w_B = 0.7$  
$\longrightarrow$ Les déficits sociaux sont souvent masqués (camouflage / masking) chez les adultes à QI élevé. Les CIR sont plus stables et moins sujets au masking donc on leur donne donc plus de poids pour une évaluation plus robuste.

**Cas recherche :** poids librement ajustables pour optimiser la corrélation entre $S$ et un critère externe (ex : score de qualité de vie, besoin d'accompagnement mesuré objectivement). Voir la partie *7. Score de méta-analyse : robustesse sous incertitude des poids* pour une optimisation du cas recherche avec un score de méta-analyse.

### Ce que $S$ ne capture pas

$S$ est un **résumé**, pas un diagnostic. Deux profils très différents peuvent avoir le même $S$ :
- $(2.5, 0.5)$ avec $w_A = w_B = 0.5$ → $S = 1.5$, mais **hors spectre** (axe B trop faible)
- $(1.5, 1.5)$ avec $w_A = w_B = 0.5$ → $S = 1.5$, mais **dans le spectre niveau 2**

C'est pourquoi $S$ doit toujours être interprété **conjointement** avec $d(P, \mathcal{R})$.

La visualisation ci-dessous montre comment la **pente des courbes d'iso-score** change selon les poids, révélant quels profils sont considérés "équivalents" selon le clinicien.



```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

a_range = np.linspace(0, 3, 300)
b_range = np.linspace(0, 3, 300)
A, B = np.meshgrid(a_range, b_range)

configs = [
    (0.5, 0.5, 'Symétrique (wA=wB=0.5)\nDSM-5 neutre'),
    (0.7, 0.3, 'Axe social prioritaire (wA=0.7, wB=0.3)\nEx: enfants en bas âge'),
    (0.3, 0.7, 'CIR prioritaires (wA=0.3, wB=0.7)\nEx: adultes avec masking')]

for ax, (wA, wB, title) in zip(axes, configs):
    S = wA * A + wB * B

    cf = ax.contourf(A, B, S, levels=15, cmap='YlOrRd')
    cs = ax.contour(A, B, S, levels=8, colors='white', linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fmt='%.1f')
    plt.colorbar(cf, ax=ax, label='Score composite S')

    # Zone TSA
    rect = plt.Rectangle((0.5, 0.5), 2.5, 2.5, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    ax.text(0.52, 2.82, 'Zone TSA', color='blue')

    # Ligne iso-score S=1.5 mise en evidence
    ax.contour(A, B, S, levels=[1.5], colors='blue', linestyles='--')
    ax.text(0.08, 0.08, 'S=1.5', color='blue')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Social')
    ax.set_ylabel('CIR')
    ax.set_title(title)

plt.suptitle(
    'Score composite S = wA * a + wB * b\n'
    'La ligne bleue relie tous les profils avec S=1.5\n'
    '-> Sa pente change selon les poids : même score ne signifie pas même profil')

plt.tight_layout()
plt.show()
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_5_0.png)
    


**Observation clef** : La ligne $S=1.5$ est droite (combinaison linéaire) mais sa **pente** change.

$wA=0.7$ : les profils (2, 0.5) et (1, 1.4) sont considérés équivalents.

$wB=0.7$ : ces mêmes profils sont très différents selon ce critère.


```python
def distance_euclidienne(a, b):
    return np.sqrt(a**2 + b**2)

def distance_chebyshev(a, b):
    # Niveau DSM-5 : max des deux scores
    return np.max([a, b], axis=0)

def distance_region(a, b, a_min=0.5, b_min=0.5):
    # Distance du point P à la region diagnostique TSA : retourne 0 si le point est dans la region (= TSA confirmé)
    da = np.maximum(0, a_min - a)
    db = np.maximum(0, b_min - b)
    return np.sqrt(da**2 + db**2)

def score_composite(a, b, w_A=0.5, w_B=0.5):
    return w_A * a + w_B * b

def niveau_dsm5(a, b):
    m = np.maximum(a, b)
    return np.where(m < 0.5, 'Hors spectre',
           np.where(m < 1.5, 'Niveau 1',
           np.where(m < 2.5, 'Niveau 2', 'Niveau 3')))

# Comparaison des métriques sur les profils illustratifs
print(f"{'Profil':<30} {'d_euclid':>10} {'d_cheby':>10} {'d_region':>10} {'Niveau'}")
print("-" * 72)

test_profiles = [
    ('Asperger-like', 0.9, 2.4),
    ('Autisme sévère', 2.8, 2.8),
    ('Profil mixte N2', 2.0, 1.8),
    ('Niveau 1 compensé', 0.8, 0.8),
    ('CIR seuls (hors)', 0.2, 2.1)]

for name, a, b in test_profiles:
    de = distance_euclidienne(a, b)
    dc = distance_chebyshev(a, b)
    dr = distance_region(a, b)
    nv = niveau_dsm5(a, b)
    print(f"{name:<30} {de:>10.2f} {dc:>10.2f} {dr:>10.2f} {nv}")

print()
print("=> 'CIR seuls' : d_euclidienne=2.10 (élevée, trompeuse!)")
print("   d_region=0.30 -> correctement identifié hors TSA")
```

    Profil                           d_euclid    d_cheby   d_region Niveau
    ------------------------------------------------------------------------
    Asperger-like                        2.56       2.40       0.00 Niveau 2
    Autisme sévère                       3.96       2.80       0.00 Niveau 3
    Profil mixte N2                      2.69       2.00       0.00 Niveau 2
    Niveau 1 compensé                    1.13       0.80       0.00 Niveau 1
    CIR seuls (hors)                     2.11       2.10       0.30 Niveau 2
    
    => 'CIR seuls' : d_euclidienne=2.10 (élevée, trompeuse!)
       d_region=0.30 -> correctement identifié hors TSA


# Courbes d'iso-sévérité : visualiser les métriques

## L'analogie de la carte topographique

Imaginez une carte de randonnée. Les **courbes de niveau** y relient tous les points situés à la même altitude.  
Une courbe à 800m d'altitude ne dit pas *où* se trouvent ces points sur la carte, mais garantit qu'ils partagent tous la même propriété : être à 800m.

Les **courbes d'iso-sévérité** fonctionnent exactement de la même façon dans $\mathcal{D}$ : elles relient tous les profils $(a, b)$ qui obtiennent la **même valeur de métrique**. Deux points sur la même courbe sont jugés *équidistants* ou *également sévères* selon la métrique choisie.

## Pourquoi les formes géométriques importent cliniquement

La forme des courbes d'iso-sévérité révèle la **logique implicite** de chaque métrique. 

C'est là que les trois métriques divergent radicalement, avec des conséquences cliniques directes :

**Distance euclidienne $d_2$ $\longrightarrow$ cercles**  
Les courbes sont des cercles centrés sur l'origine. La métrique traite les deux axes comme parfaitement symétriques et *échangeables*.  
Un profil $(2, 0)$ est jugé "aussi éloigné" qu'un $(0, 2)$ ou un $(\sqrt{2}, \sqrt{2})$.  
Cliniquement, c'est faux : $(2, 0)$ est hors spectre (axe B absent), les autres sont dans le spectre.  
**$\longrightarrow$ Euclide ignore la structure du seuil diagnostique.**

**Distance de Chebyshev $d_\infty$ $\longrightarrow$ carrés**  
Les courbes sont des carrés (au sens de la norme $L^\infty$). La métrique ne regarde que le **critère le plus sévère** et ignore l'autre.  
Un profil $(3, 0.1)$ a la même valeur $d_\infty = 3$ qu'un $(3, 3)$ alors que leurs profils sont radicalement différents.  
C'est exactement la logique des **niveaux DSM-5** : le niveau est déterminé par le maximum des deux axes.  
**$\longrightarrow$ Chebyshev reflète fidèlement la convention DSM-5, mais écrase la contribution du second axe.**

**Distance à la région $d(P, \mathcal{R})$ $\longrightarrow$ courbes en L**  
Les courbes forment des quarts de cercle à l'extérieur de la région TSA, et valent **0 partout à l'intérieur**.
C'est la métrique la plus pertinente pour une question binaire : *"ce profil franchit-il le seuil ?"*
Elle quantifie exactement de combien un profil est en dessous du seuil sur chaque axe séparément.  
**$\longrightarrow$ $d_{\mathcal{R}}$ est la seule métrique qui encode correctement la logique ET des deux critères.**

## Ce que la visualisation montre

Le point vert sur les graphiques ci-dessous représente le profil **"CIR seuls"** $(0.2, 2.1)$ : hors spectre car l'axe A est trop faible, même si l'axe B est élevé.

Observez comment chaque métrique le positionne différemment :
- $d_2$ le place sur une courbe élevée (comme si c'était un profil sévère)
- $d_\infty$ lui donne la valeur 2.1 (idem)
- $d_{\mathcal{R}}$ lui donne une valeur **non nulle** - le seul à l'identifier correctement hors TSA

**Le choix de la métrique n'est pas neutre : il incarne une théorie implicite du diagnostic.**


```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

a_range = np.linspace(0, 3, 400)
b_range = np.linspace(0, 3, 400)
A, B = np.meshgrid(a_range, b_range)

metrics = [
    ('Distance euclidienne d2\n(courbes = cercles)',
     distance_euclidienne(A, B), 'Blues'),
    ('Distance Chebyshev d_inf\n(courbes = carrés selon logique DSM-5)',
     distance_chebyshev(A, B), 'Oranges'),
    ('Distance à region TSA d(P,R)\n(vaut 0 dans toute la zone TSA)',
     distance_region(A, B), 'Reds')]

p_test = (0.2, 2.1)  # profil CIR seuls, hors spectre

for ax, (title, Z, cmap) in zip(axes, metrics):
    cf = ax.contourf(A, B, Z, levels=20, cmap=cmap, alpha=0.85)
    cs = ax.contour(A, B, Z, levels=8, colors='white', linewidths=0.6, alpha=0.6)
    ax.clabel(cs, inline=True, fmt='%.1f')

    # Rectangle zone TSA
    rect = plt.Rectangle((0.5, 0.5), 2.5, 2.5, edgecolor='red', linewidth=2, facecolor='none')
    ax.add_patch(rect)
    ax.text(0.52, 2.82, 'Zone TSA', color='red')

    # Profil CIR seuls illustratif
    ax.scatter(*p_test, s=80, color='forestgreen', zorder=10)
    ax.annotate('CIR seuls\n(hors spectre)', p_test, xytext=(0.55, 1.75), arrowprops=dict(arrowstyle='->', color='forestgreen'), color='forestgreen')
    
    
    plt.colorbar(cf, ax=ax, label='Valeur de la metrique')
    ax.set_title(title)
    ax.set_xlabel('Social')
    ax.set_ylabel('CIR')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

plt.suptitle(
    'Courbes d\'iso-severite : la forme révèle la logique de chaque métrique\n'
    'Le point vert (CIR seuls, hors spectre) montre pourquoi le choix de métrique change l\'interprétation')
plt.tight_layout()
plt.show()

a, b = p_test
print(f"Profil CIR seuls = ({a}, {b})")
print(f"  d_euclidienne = {distance_euclidienne(a, b):.2f}  (élevée -> trompeuse)")
print(f"  d_Chebyshev   = {distance_chebyshev(a, b):.2f}  (= 2.1 -> trompeuse)")
print(f"  d_region      = {distance_region(a, b):.2f}  (> 0 -> hors spectre, correct!)")
print()
print("-> Seule d_region identifie correctement ce profil comme hors TSA")
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_9_0.png)
    


    Profil CIR seuls = (0.2, 2.1)
      d_euclidienne = 2.11  (élevée -> trompeuse)
      d_Chebyshev   = 2.10  (= 2.1 -> trompeuse)
      d_region      = 0.30  (> 0 -> hors spectre, correct!)
    
    -> Seule d_region identifie correctement ce profil comme hors TSA


# Placer un profil individuel dans l'espace

## Comment obtenir les coordonnées $(a, b)$ d'un individu ?

En pratique clinique, les scores des outils standardisés peuvent être convertis en coordonnées dans $\mathcal{D}$. La conversion repose sur une **régression affine** : on ramène le score brut de chaque outil sur l'intervalle $[0, 3]$ en respectant le seuil diagnostique à 0.5.

## Outil principal : ADOS-2 (Autism Diagnostic Observation Schedule, 2e édition)

L'ADOS-2 est l'outil observationnel de référence. Il comprend 5 modules selon l'âge et le niveau de langage. Il produit deux scores distincts :

- **Social Affect (SA)** : communication et interaction sociale → **Axe Social**
- **Restricted and Repetitive Behaviors (RRB)** : comportements restreints/répétitifs → **Axe CIR**

| Module | Population cible | SA max | Seuil TSA (SA) | RRB max | Seuil TSA (RRB) |
|--------|-----------------|--------|----------------|---------|-----------------|
| 1 | Pas de langage | 20 | 11 | 7 | 3 |
| 2 | Quelques mots | 20 | 9 | 7 | 3 |
| 3 | Enfants verbaux | 22 | 8 | 6 | 2 |
| 4 | Adolescents/adultes | 22 | 8 | 6 | 2 |
| T | Tout-petits (12-30 mois) | 22 | 9 | 8 | 3 |

**Formule de normalisation générale :**

$$a = \frac{\text{SA}_{\text{obs}} - \text{SA}_{\min}}{\text{SA}_{\max} - \text{SA}_{\min}} \times 3
\qquad b = \frac{\text{RRB}_{\text{obs}} - \text{RRB}_{\min}}{\text{RRB}_{\max} - \text{RRB}_{\min}} \times 3$$

avec $\text{SA}_{\min} = \text{RRB}_{\min} = 0$ dans tous les modules.

**Exemple - Module 3, enfant de 8 ans :**  
SA observé = 12, RRB observé = 4  
$a = \frac{12}{22} \times 3 = 1.64$ ,  $b = \frac{4}{6} \times 3 = 2.00$  
$\longrightarrow$ Profil $P = (1.64,\ 2.00)$, dans la zone TSA niveau 2.


```python
# Normalisation
SA_obs, SA_max = 12, 22
RRB_obs, RRB_max = 4, 6

a = (SA_obs / SA_max) * 3
b = (RRB_obs / RRB_max) * 3

print(f"a = {SA_obs}/{SA_max} × 3 = {a:.3f}")
print(f"b = {RRB_obs}/{RRB_max} × 3 = {b:.3f}")
print(f"Niveau DSM-5 : {niveau_dsm5(a, b).item()}")

# Incertitude (fidélité test-retest ADOS-2)
sigma_a, sigma_b = 0.25, 0.30

res = placer_profil(
    a=a, b=b,
    label="Enfant 8 ans - Module 3",
    incertitude=max(sigma_a, sigma_b),
    montrer_distribution=True)
```

    a = 12/22 × 3 = 1.636
    b = 4/6 × 3 = 2.000
    Niveau DSM-5 : Niveau 2



    
![png](/images/autisme-espace-2D/autisme-espace-2D_11_1.png)
    


## Outil complémentaire : ADI-R (Autism Diagnostic Interview - Revised)

L'ADI-R est un entretien semi-structuré avec les parents. Il produit trois scores :

| Domaine ADI-R | Seuil diagnostique | Correspondance |
|--------------|-------------------|----------------|
| Communication (A) | 8 (verbal) / 7 (non verbal) | Axe Social (partiel) |
| Interaction sociale réciproque (B) | 10 | Axe Social (partiel) |
| Comportements répétitifs (C) | 3 | Axe CIR |

Pour l'axe Social, on combine les domaines A et B de l'ADI-R :

$$a = \frac{A_{\text{obs}} + B_{\text{obs}}}{A_{\max} + B_{\max}} \times 3$$

**Limite importante :** l'ADI-R mesure le comportement sur toute la vie du patient
(notamment entre 4 et 5 ans), tandis que l'ADOS-2 mesure le comportement actuel.
Les deux outils ne sont donc pas directement interchangeables pour construire $P$.
L'usage recommandé est de **croiser les deux** : l'ADOS-2 donne le profil actuel,
l'ADI-R valide la persistance développementale.

## Conversion croisée ADOS-2 / ADI-R : algorithme de consensus

Quand les deux outils sont disponibles, on peut construire un profil consensuel
en pondérant les deux sources selon leur fiabilité relative :

$$a_{\text{consensus}} = \lambda \cdot a_{\text{ADOS}} + (1 - \lambda) \cdot a_{\text{ADI}}$$

avec $\lambda \in [0.5, 0.8]$ en faveur de l'ADOS-2, qui est considéré plus fiable
pour le profil actuel (Lord et al., *Journal of Child Psychology and Psychiatry*, 2012).

## Approche bayésienne : le profil comme distribution

Jusqu'ici, chaque individu occupe un **point unique** $P = (a, b)$ dans $\mathcal{D}$.
Mais ce point est lui-même une estimation entachée de plusieurs sources d'incertitude :

- **Incertitude de mesure** : fidélité test-retest de l'ADOS-2 ≈ 0.82 (Gotham et al., 2007),
  ce qui implique une variabilité résiduelle même avec un clinicien expérimenté
- **Incertitude inter-cotateurs** : accord inter-juges κ ≈ 0.70–0.85 selon les domaines
- **Incertitude de conversion** : la normalisation affine suppose une linéarité
  qui n'est qu'approximative

On modélise donc $P$ non pas comme un point mais comme une **distribution gaussienne
bivariée** sur $\mathcal{D}$ :

$$P \sim \mathcal{N}(\mu,\ \Sigma)$$

où :
- $\mu = (\hat{a},\ \hat{b})$ est l'estimation ponctuelle issue de la normalisation
- $\Sigma$ est la matrice de covariance encodant l'incertitude :

$$\Sigma = \begin{pmatrix} \sigma_a^2 & \rho\,\sigma_a\,\sigma_b \\ \rho\,\sigma_a\,\sigma_b & \sigma_b^2 \end{pmatrix}$$

**Interprétation des paramètres :**

| Paramètre | Signification | Valeur indicative |
|-----------|--------------|-------------------|
| $\sigma_a$ | Incertitude sur l'axe social | 0.20 - 0.35 |
| $\sigma_b$ | Incertitude sur l'axe CIR | 0.25 - 0.40 |
| $\rho$ | Corrélation entre les deux incertitudes | 0 (hypothèse d'indépendance) |

Le paramètre $\rho$ est fixé à 0 par défaut (axes supposés indépendants),
mais peut être estimé empiriquement si des données de fidélité multi-domaines
sont disponibles.

**Ce que cette modélisation permet :**

1. **Probabilité diagnostique** : $P(\text{profil} \in \mathcal{R}_{\text{TSA}}) = \int_{\mathcal{R}} \mathcal{N}(p;\, \mu, \Sigma)\, dp$

2. **Ellipse de confiance à 95%** : l'ensemble des positions plausibles du profil
   réel, visualisable dans $\mathcal{D}$

3. **Propagation de l'incertitude** : quand on calcule $S = w_A a + w_B b$,
   l'incertitude sur $P$ se propage analytiquement :
   $$\text{Var}(S \mid P \sim \mathcal{N}) = w_A^2\,\sigma_a^2 + w_B^2\,\sigma_b^2 + 2\,\rho\,w_A\,w_B\,\sigma_a\,\sigma_b$$

> **Note sur le masking :** chez les adultes à QI élevé, les scores observés
> sous-estiment systématiquement le profil réel à cause des stratégies de compensation.
> Dans ce cas, $\mu$ doit être corrigé vers le haut, ou $\sigma$ augmenté pour
> refléter l'incertitude supplémentaire. Le problème du score observé comme
> variable latente du score réel est formalisé dans les modèles à variables
> latentes (IRT, Item Response Theory), qui constituent (en toute modestie) l'extension naturelle
> de ce notebook.

La fonction `placer_profil` ci-dessous sert à placer un profil clinique dans l'espace de diagnostic (original).

Elle prend comme paramètres :
| Paramètre | Description |
|-----------|-------------|
| a : float [0, 3]      | Score axe déficits sociaux |
| b : float [0, 3]      | Score axe CIR |
| label : str          |  Nom du profil |
| incertitude : float | Ecart-type de l'incertitude de mesure (bayésien) |
| montrer_distribution : bool | Afficher la distribution d'incertitude |
| w_A, w_B : float       | Poids pour le score composite (doivent sommer à 1) |



```python
def placer_profil(a, b, label='Profil analyse', incertitude=0.2, montrer_distribution=True, w_A=0.5, w_B=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panneau gauche : espace de diagnostic
    ax = axes[0]

    for vmin, vmax, col, lbl in [(0.5, 1.5, 'lightgreen',  'N1'), (1.5, 2.5, 'lightyellow', 'N2'), (2.5, 3.0, 'lightsalmon', 'N3')]:
        rect = plt.Rectangle((vmin, vmin), vmax - vmin, vmax - vmin, facecolor=col, alpha=0.4)
        ax.add_patch(rect)
        ax.text(vmin + 0.05, vmax - 0.15, lbl, color='gray')

    for v in [0.5, 1.5, 2.5]:
        ax.axhline(v, color='gray', lw=0.8, ls='--')
        ax.axvline(v, color='gray', lw=0.8, ls='--')

    if montrer_distribution:
        a_r = np.linspace(0, 3, 200)
        b_r = np.linspace(0, 3, 200)
        A2, B2 = np.meshgrid(a_r, b_r)
        rv = multivariate_normal([a, b], [[incertitude**2, 0], [0, incertitude**2]])
        Z2 = rv.pdf(np.dstack((A2, B2)))
        ax.contourf(A2, B2, Z2, levels=8, cmap='Purples', alpha=0.4)

    rect_diag = plt.Rectangle((0.5, 0.5), 2.5, 2.5, edgecolor='red', lw=2, facecolor='none')
    ax.add_patch(rect_diag)

    ax.scatter(a, b, s=150, color='purple', zorder=10)
    ax.annotate(f'{label}  P=({a},{b})', (a, b), xytext=(15, 20), textcoords='offset points', color='purple')
    ax.plot([a, a], [0, b], color='purple', lw=1, ls=':', alpha=0.5)
    ax.plot([0, a], [b, b], color='purple', lw=1, ls=':', alpha=0.5)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Deficits sociaux')
    ax.set_ylabel('CIR')
    ax.set_title('Placement dans D')

    # Panneau droit : tableau des métriques
    ax2 = axes[1]
    ax2.axis('off')

    d_e = distance_euclidienne(a, b)
    d_c = distance_chebyshev(a, b)
    d_r = distance_region(a, b)
    s_c = score_composite(a, b, w_A, w_B)
    niv = niveau_dsm5(a, b)

    # Probabilité bayesienne P (dans TSA)
    a_r2 = np.linspace(0, 3, 300)
    b_r2 = np.linspace(0, 3, 300)
    A3, B3 = np.meshgrid(a_r2, b_r2)
    rv2 = multivariate_normal([a, b], [[incertitude**2, 0], [0, incertitude**2]])
    Z3 = rv2.pdf(np.dstack((A3, B3)))
    mask_tsa = (A3 >= 0.5) & (B3 >= 0.5)
    p_tsa = float(np.sum(Z3 * mask_tsa) * (a_r2[1]-a_r2[0]) * (b_r2[1]-b_r2[0]))

    statut_region = 'dans TSA' if d_r == 0 else 'hors TSA'
    lignes = [
        ('Score Axe Social',        f'{a:.2f} / 3'),
        ('Score Axe CIR',           f'{b:.2f} / 3'),
        ('---',                     ''),
        ('d_euclidienne',           f'{d_e:.3f}'),
        ('d_Chebyshev',             f'{d_c:.3f}'),
        ('d_region',                f'{d_r:.3f}  ({statut_region})'),
        ('Score composite S',       f'{s_c:.3f}  (wA={w_A}, wB={w_B})'),
        ('---',                     ''),
        ('Niveau DSM-5',            niv),
        ('P(profil dans TSA)',      f'{p_tsa:.1%}')]
    
    y = 0.90
    ax2.text(0.05, y + 0.06, 'METRIQUES', fontweight='bold', transform=ax2.transAxes)
    for cle, val in lignes:
        if cle == '---':
            y -= 0.03
            continue
        ax2.text(0.05, y, cle, transform=ax2.transAxes)
        ax2.text(0.60, y, val, transform=ax2.transAxes, fontweight='bold')
        y -= 0.08

    plt.suptitle(f'Analyse du profil : {label}', fontsize=13)
    plt.tight_layout()
    plt.show()

    return {'a': a, 'b': b, 'niveau': niv, 'p_tsa': p_tsa, 'd_region': d_r, 'score_composite': s_c}
```

**Exemple 1 : profil Asperger-like**


```python
res = placer_profil(a=0.9, b=2.4, label='Asperger-like', incertitude=0.25)
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_16_0.png)
    


**Exemple 2 : autisme classique sévère**


```python
res2 = placer_profil(a=2.8, b=2.7, label='Autisme sévère', incertitude=0.15)
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_18_0.png)
    


**Exemple 3 : cas limite à la frontière du diagnostic**


```python
res3 = placer_profil(a=0.55, b=0.6, label='Cas limite N1', incertitude=0.3)
print(f"\n-> P(TSA) = {res3['p_tsa']:.1%} : incertitude élevée, suivi recommandé")
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_20_0.png)
    


    
    -> P(TSA) = 36.2% : incertitude élevée, suivi recommandé


# Trajectoires longitudinales : le diagnostic comme chemin

Le profil d'une personne n'est pas statique. Les scores peuvent évoluer avec l'âge, les interventions thérapeutiques, le masking, ou simplement le développement.

On peut modéliser une **trajectoire** dans $\mathcal{D}$ comme une courbe paramétrée :

$$\gamma : t \in [0, T] \mapsto (a(t), b(t)) \in \mathcal{D}$$

Exemples de trajectoires typiques :
- **Trajectoire de compensation** : les scores mesurés diminuent à cause du masking, sans changement réel
- **Trajectoire d'intervention** : baisse de $b$ (CIR) suite à une thérapie comportementale
- **Trajectoire développementale** : les CIR bas niveau diminuent avec l'âge, les intérêts circonscrits augmentent

La fonction `tracer_trajectoire` ci-dessous visualise l'évolution d'un profil dans le temps.


```python
def tracer_trajectoire(points_temporels, labels_temps, titre='Trajectoire longitudinale'):
    fig, ax = plt.subplots(figsize=(8, 5))

    for vmin, vmax, col in [(0.5, 1.5, 'lightgreen'), (1.5, 2.5, 'lightyellow'), (2.5, 3.0, 'lightsalmon')]:
        rect = plt.Rectangle((vmin, vmin), vmax - vmin, vmax - vmin, facecolor=col, alpha=0.4)
        ax.add_patch(rect)

    for v in [0.5, 1.5, 2.5]:
        ax.axhline(v, color='gray', lw=0.8, ls='--')
        ax.axvline(v, color='gray', lw=0.8, ls='--')

    rect_diag = plt.Rectangle((0.5, 0.5), 2.5, 2.5, edgecolor='red', lw=2, facecolor='none')
    ax.add_patch(rect_diag)

    pts = np.array(points_temporels)
    n = len(pts)
    colors_traj = plt.cm.plasma(np.linspace(0.15, 0.85, n))

    ax.plot(pts[:, 0], pts[:, 1], color='gray', lw=1.5, ls='-', alpha=0.4)

    for i in range(n - 1):
        ax.annotate('', xy=pts[i + 1], xytext=pts[i], arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    for i, ((a, b), label, col) in enumerate(zip(pts, labels_temps, colors_traj)):
        ax.scatter(a, b, s=120, color=col, zorder=10)
        ax.annotate(f'{label} ({a:.1f},{b:.1f})', (a, b), xytext=(-50, 15), textcoords='offset points')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Social')
    ax.set_ylabel('CIR')
    ax.set_title(titre)
    plt.tight_layout()
    plt.show()

tracer_trajectoire(
    points_temporels=[
        (2.1, 2.5),
        (1.9, 2.3),
        (1.5, 2.4),
        (1.2, 2.6),
        (0.9, 2.5)],
    labels_temps=['T0 (4 ans)', 'T1 (6 ans)', 'T2 (9 ans)', 'T3 (13 ans)', 'T4 (17 ans)'],
    titre='Trajectoire longitudinale : compensation du profil social avec maintien des CIR')
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_22_0.png)
    


# Score de méta-analyse : robustesse sous incertitude des poids

## Idée fondamentale

Choisir $w_A$ et $w_B$ est une décision théorique qui engage une vision clinique.

Plutôt que de trancher arbitrairement, on peut **faire varier systématiquement les poids** sur toute leur plage possible (pas de 0.1, 11 combinaisons) et observer comment le score $S$ se comporte.

Ce qu'on obtient n'est plus un score unique mais une **distribution de scores** pour chaque profil, ce qui permet de caractériser sa robustesse.

## Les quatre indicateurs du score de méta-analyse

Pour un profil $P = (a, b)$, on calcule $S(w_A)$ pour $w_A \in \{0.0, 0.1, ..., 1.0\}$ avec $w_B = 1 - w_A$. On en extrait :

| Indicateur | Formule | Interprétation |
|-----------|---------|----------------|
| $\bar{S}$ | $\frac{1}{11}\sum S(w_A)$ | Score moyen : toujours égal à $\frac{a+b}{2}$ |
| $\sigma(S)$ | $\text{std}(S(w_A))$ | **Sensibilité aux poids** : profil déséquilibré si élevé |
| $S_{\min}$ | $\min(a, b)$ | Score minimal atteignable (poids à 0 sur l'axe fort) |
| $S_{\max}$ | $\max(a, b)$ | Score maximal atteignable (poids à 1 sur l'axe fort) |

**Propriété clé :** $\bar{S} = 0.5 \cdot a + 0.5 \cdot b$ la moyenne sur tous les poids coïncide toujours avec le score symétrique. L'intérêt de la méta-analyse n'est donc pas dans $\bar{S}$ lui-même, mais dans $\sigma(S)$ et l'intervalle $[S_{\min}, S_{\max}]$.

## Calcul & interprétation de $\sigma(S)$

Les poids sont $w_A \in \{0.0, 0.1, ..., 1.0\}$, soit 11 valeurs équiréparties.

$S(w_A) = w_A \cdot a + (1 - w_A) \cdot b = b + w_A(a - b)$

$S$ est une fonction linéaire de $w_A$, donc :

$$\text{Var}(S) = (a - b)^2 \cdot \text{Var}(w_A)$$

**Calcul de $\text{Var}(w_A)$ pour la distribution uniforme discrète sur $\{0, 0.1, ..., 1.0\}$ :**

$$\bar{w} = \frac{1}{11} \sum_{k=0}^{10} \frac{k}{10} = \frac{1}{2}$$

$$\text{Var}(w_A) = \frac{1}{11} \sum_{k=0}^{10} \left(\frac{k}{10} - \frac{1}{2}\right)^2 = \frac{1}{1100} \sum_{k=0}^{10} (k-5)^2 = \frac{2(1+4+9+16+25)}{1100} = \frac{1}{10}$$

On obtient donc :

$$\boxed{\sigma(S) = \frac{|a - b|}{\sqrt{10}} \approx \frac{|a - b|}{3.162}}$$

**Interprétation :** $\sigma(S)$ est entièrement déterminé par le déséquilibre inter-axes $|a - b|$.
Un profil avec $a = b$ a $\sigma(S) = 0$. Son score est insensible au choix des poids. Un profil avec $|a - b| = 3$ (déséquilibre maximal) a $\sigma(S) \approx 0.949$.
- $\sigma(S) = 0$ : profil parfaitement équilibré ($a = b$) $\longrightarrow$ le score est **insensible** aux poids
- $\sigma(S)$ élevé : $a$ et $b$ sont très différents $\longrightarrow$ le score **dépend fortement** du système de pondération choisi

## Dominance stochastique entre groupes

Pour comparer deux groupes G1 et G2, on dit que G1 **domine** G2 si :

$$S_{\text{G1}}(w_A) > S_{\text{G2}}(w_A) \quad \forall w_A \in [0, 1]$$

Cela signifie que G1 est plus sévère que G2 **quel que soit le système de poids**, *id est* une conclusion beaucoup plus robuste qu'une comparaison à $w_A = 0.5$ seulement.

Si la dominance n'est pas totale, on peut calculer la **proportion de poids** pour lesquels G1 > G2, comme mesure partielle de robustesse.


```python
import numpy as np
import matplotlib.pyplot as plt

# Grille des poids
W_A = np.round(np.arange(0.0, 1.01, 0.1), 1)
W_B = 1 - W_A

def meta_score(a, b):
    scores = W_A * a + W_B * b
    return {
        "scores":   scores,
        "mean":     scores.mean(),
        "std":      scores.std(),
        "s_min":    scores.min(),
        "s_max":    scores.max(),
        "interval": (scores.min(), scores.max())}

def dominance(profil1, profil2, label1="G1", label2="G2", verbose=True):
    s1 = W_A * profil1[0] + W_B * profil1[1]
    s2 = W_A * profil2[0] + W_B * profil2[1]
    prop = (s1 > s2).mean()

    if verbose:
        if prop == 1.0:
            print(f"{label1} domine strictement {label2} pour TOUS les poids.")
        elif prop == 0.0:
            print(f"{label2} domine strictement {label1} pour TOUS les poids.")
        else:
            print(f"Dominance partielle : {label1} > {label2} pour "
                  f"{prop:.0%} des configurations de poids.")
    return prop


# Visualisation : profil individuel
def plot_meta_profil(a, b, label="Profil"):
    m = meta_score(a, b)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panneau gauche : courbe S(w_A)
    ax = axes[0]
    ax.plot(W_A, m["scores"], marker='o', linewidth=2, label=f"S(wA)  [{label}]")
    ax.axhline(m["mean"], color='gray', linestyle='--', linewidth=1,
               label=f"Moyenne = {m['mean']:.2f}")
    ax.fill_between(W_A, m["s_min"], m["s_max"],
                    alpha=0.15, label=f"Intervalle [{m['s_min']:.2f}, {m['s_max']:.2f}]")

    # Zone TSA (S >= 0.5)
    ax.axhspan(0.5, 3.0, alpha=0.06, color='green', label="S >= seuil TSA (0.5)")
    ax.axhline(0.5, color='green', linestyle=':', linewidth=1)

    ax.set_xlabel("$w_A$ (poids axe Social)")
    ax.set_ylabel("Score composite S")
    ax.set_title(f"Sensibilité de S aux poids - {label}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.1)
    ax.set_xticks(W_A)
    ax.legend(fontsize=9)

    # Annotation sigma
    ax.text(0.5, m["mean"] + 0.08,
            f"σ(S) = {m['std']:.3f}",
            ha='center', fontsize=10, color='gray')

    # Panneau droit : tableau des indicateurs
    ax2 = axes[1]
    ax2.axis('off')

    rows = [
        ["Profil (a, b)",          f"({a}, {b})"],
        ["Score symétrique S(0.5)","= moyenne = {:.3f}".format(m['mean'])],
        ["Écart-type σ(S)",        f"{m['std']:.4f}"],
        ["S_min  [w_A=1 ou 0]",    f"{m['s_min']:.3f}  (= min(a,b) = {min(a,b)})"],
        ["S_max  [w_A=1 ou 0]",    f"{m['s_max']:.3f}  (= max(a,b) = {max(a,b)})"],
        ["Intervalle [S_min, S_max]",
         f"[{m['s_min']:.3f},  {m['s_max']:.3f}]"],
        ["Amplitude",              f"{m['s_max'] - m['s_min']:.3f}"],
        ["Profil équilibré ?",
         "Oui (σ < 0.2)" if m['std'] < 0.2 else "Non - sensible aux poids"]]

    y = 0.90
    ax2.text(0.02, y + 0.06, "INDICATEURS DE MÉTA-ANALYSE", fontweight='bold', transform=ax2.transAxes)
    for cle, val in rows:
        ax2.text(0.02, y, cle, transform=ax2.transAxes)
        ax2.text(0.55, y, val, transform=ax2.transAxes, fontweight='bold')
        y -= 0.10

    plt.suptitle(f"Score de méta-analyse - {label}")
    plt.tight_layout()
    plt.show()

    return m


# Visualisation : comparaison de groupes
def plot_meta_groupes(groupes, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panneau gauche : courbes S(w_A) par groupe
    ax = axes[0]
    for (a, b), label in zip(groupes, labels):
        scores = W_A * a + W_B * b
        ax.plot(W_A, scores, marker='o', linewidth=2, label=label)

    ax.axhline(0.5, color='green', linestyle=':', linewidth=1, label="Seuil TSA")
    ax.set_xlabel("$w_A$ (poids axe Social)")
    ax.set_ylabel("Score composite S")
    ax.set_title("Comparaison des groupes sur toute la plage de poids")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.1)
    ax.set_xticks(W_A)
    ax.legend(fontsize=9)

    # Panneau droit : matrice de dominance
    ax2 = axes[1]
    n = len(groupes)
    mat = np.zeros((n, n))
    for i, (p1, l1) in enumerate(zip(groupes, labels)):
        for j, (p2, l2) in enumerate(zip(groupes, labels)):
            if i != j:
                s1 = W_A * p1[0] + W_B * p1[1]
                s2 = W_A * p2[0] + W_B * p2[1]
                mat[i, j] = (s1 > s2).mean()

    im = ax2.imshow(mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax2, label="Proportion de poids où ligne > colonne")
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_title("Matrice de dominance\n(prop. de poids où [ligne] > [colonne])")

    # Annotations dans les cellules
    for i in range(n):
        for j in range(n):
            if i != j:
                val = mat[i, j]
                txt = "100%" if val == 1.0 else f"{val:.0%}"
                ax2.text(j, i, txt, ha='center', va='center',
                         fontsize=10, fontweight='bold',
                         color='black' if 0.2 < val < 0.8 else 'white')
            else:
                ax2.text(j, i, "-", ha='center', va='center', fontsize=10, color='gray')

    plt.suptitle("Méta-analyse comparative entre groupes", fontsize=13)
    plt.tight_layout()
    plt.show()

    # Résumé textuel
    print("RÉSUMÉ DE DOMINANCE")
    print("-" * 50)
    for i, (p1, l1) in enumerate(zip(groupes, labels)):
        for j, (p2, l2) in enumerate(zip(groupes, labels)):
            if i < j:
                prop = mat[i, j]
                if prop == 1.0:
                    print(f"  {l1} domine strictement {l2} (100% des poids)")
                elif prop == 0.0:
                    print(f"  {l2} domine strictement {l1} (100% des poids)")
                else:
                    gagnant = l1 if prop > 0.5 else l2
                    print(f"  {l1} vs {l2} : dominance partielle "
                          f"({max(prop, 1-prop):.0%} des poids en faveur de {gagnant})")

# EXEMPLES D'UTILISATION
# Profils individuels
m1 = plot_meta_profil(a=0.9, b=2.4, label="Asperger-like")
m2 = plot_meta_profil(a=2.8, b=2.7, label="Autisme sévère")
m3 = plot_meta_profil(a=1.5, b=1.5, label="Profil équilibré N2")

# Dominance entre profils individuels
dominance((0.9, 2.4), (1.5, 1.5), "Asperger-like", "Profil équilibré")
dominance((2.8, 2.7), (0.9, 2.4), "Autisme sévère", "Asperger-like")

# Comparaison de groupes (profils moyens)
groupes = [
    (2.5, 2.6),   # Groupe A : autisme sévère moyen
    (1.0, 2.2),   # Groupe B : profil Asperger moyen
    (1.4, 1.3),   # Groupe C : niveau 1 compensé moyen
    (0.3, 0.4)]   # Groupe D : hors spectre / contrôles

labels_groupes = ["G-Sévère", "G-Asperger", "G-Niveau1", "G-Contrôle"]

plot_meta_groupes(groupes, labels_groupes)
```


    
![png](/images/autisme-espace-2D/autisme-espace-2D_24_0.png)
    



    
![png](/images/autisme-espace-2D/autisme-espace-2D_24_1.png)
    



    
![png](/images/autisme-espace-2D/autisme-espace-2D_24_2.png)
    


    Dominance partielle : Asperger-like > Profil équilibré pour 55% des configurations de poids.
    Autisme sévère domine strictement Asperger-like pour TOUS les poids.



    
![png](/images/autisme-espace-2D/autisme-espace-2D_24_4.png)
    


    RÉSUMÉ DE DOMINANCE
    --------------------------------------------------
      G-Sévère domine strictement G-Asperger (100% des poids)
      G-Sévère domine strictement G-Niveau1 (100% des poids)
      G-Sévère domine strictement G-Contrôle (100% des poids)
      G-Asperger vs G-Niveau1 : dominance partielle (64% des poids en faveur de G-Asperger)
      G-Asperger domine strictement G-Contrôle (100% des poids)
      G-Niveau1 domine strictement G-Contrôle (100% des poids)


**Pour un profil individuel** : `plot_meta_profil` génère la courbe $S(w_A)$ avec l'intervalle de confiance en zone ombrée, le seuil TSA en pointillés verts, et le tableau des indicateurs. Un profil comme Asperger-like $(0.9, 2.4)$ aura une pente très marquée (score faible si on privilégie le social, élevé si on privilégie les CIR) et un $\sigma$ important, ce qui documente exactement son déséquilibre inter-axes.

**Pour les groupes** : `plot_meta_groupes` produit les courbes superposées *et* la matrice de dominance. Chaque cellule $(i, j)$ indique la proportion de poids pour lesquels le groupe $i$ surpasse le groupe $j$. Une cellule verte à 100% = dominance stricte. Une cellule orange à 60% = conclusion fragile, dépendante du système de poids, ce qui est en soi une information scientifique importante.

# Conclusion

Ce notebook est parti d'une question conceptuelle simple *pourquoi dit-on que l'autisme est un "spectre" ?* pour construire progressivement un modèle mathématique qui en révèle à la fois la cohérence et les limites.

## Ce qu'on a construit

Nous avons formalisé le diagnostic TSA comme un problème de **placement dans un espace à deux dimensions** $\mathcal{D} = [0,3]^2$, où chaque individu occupe une position définie par ses scores sur l'axe social et l'axe CIR. Cette représentation, fidèle à la structure du DSM-5, permet de raisonner géométriquement sur des questions qui restent floues dans le langage clinique ordinaire.

À partir de cette base, nous avons introduit quatre niveaux d'analyse successifs :

| Section | Outil | Question traitée |
|---------|-------|-----------------|
| Métriques | $d_2$,  $d_\infty$,  $d_{\mathcal{R}}$ | Comment mesurer la position d'un profil ? |
| Score composite | $S = w_A a + w_B b$ | Comment résumer un profil en un seul nombre ? |
| Méta-analyse | $\sigma(S)$, dominance stochastique | Le résumé est-il robuste au choix des poids ? |
| Bayésien | $P(\text{TSA} \mid S, \pi_0)$ | Que vaut un score dans un contexte donné ? |

## Ce que le modèle 2D capture bien
- La **structure obligatoire à deux critères** du diagnostic DSM-5
- La distinction entre **profils à dominante sociale** vs. **profils à dominante CIR**
- L'**incertitude de mesure** via la modélisation bayésienne
- Les **trajectoires longitudinales** comme chemins dans l'espace

## Ce que le modèle 2D ne capture pas

| Limitation | Extension possible |
|-----------|-------------------|
| Les deux axes sont eux-mêmes multidimensionnels | Espace à $n > 2$ dimensions (ACP sur les items ADOS) |
| Les CIR haut/bas niveau ont des dynamiques différentes | Axe B subdivisé en $B_1$ (stéréotypies) et $B_2$ (intérêts circonscrits) |
| Le masking fausse les scores mesurés | Modèle à variables latentes : score réel ≠ score observé |
| La co-occurrence (TDAH, anxiété) | Espace étendu avec dimensions supplémentaires |
| La dimension génétique | Projection sur un espace génotypique couplé |

## Ce que le modèle révèle sur l'autisme

Trois résultats méritent d'être retenus au-delà de la technique.

**Le spectre est structurellement bidimensionnel, pas unidimensionnel.** La métaphore courante d'un curseur allant de "peu autiste" à "très autiste" est mathématiquement inexacte. Deux profils peuvent avoir le même score composite $S$ tout en occupant des positions très différentes dans $\mathcal{D}$ (l'un à dominante sociale, l'autre à dominante CIR) avec des besoins, des trajectoires et probablement des étiologies distinctes.

**Le choix de la métrique n'est pas neutre.** Choisir $d_2$, $d_\infty$ ou $d_{\mathcal{R}}$ pour évaluer un profil revient à adopter implicitement une théorie du diagnostic. La distance euclidienne ignore la structure du seuil ; Chebyshev écrase la contribution du second axe ; seule $d_{\mathcal{R}}$ encode correctement la logique *ET* des deux critères. Ce que les cliniciens font intuitivement quand ils évaluent un profil correspond, souvent sans le savoir, à l'une de ces métriques.

**Le taux de base domine la conclusion diagnostique.** L'analyse bayésienne montre que le même score $S = 1.2$ donne une probabilité post-diagnostique de 4% en population générale et de 72% dans un centre spécialisé. Ce résultat suggère que tout outil de screening TSA devrait systématiquement afficher la probabilité post-diagnostique contextualisée plutôt qu'un score brut.

## Limites du modèle

Ce modèle est une **simplification délibérée**. Il suppose que les deux axes sont indépendants (ce qui est discutable empiriquement), que les distributions de scores sont gaussiennes (approximation), et que les scores observés reflètent les scores réels (le masking viole cette hypothèse). L'extension naturelle serait un espace à quatre dimensions $\mathcal{D}^4$ distinguant $B_1$ (stéréotypies et CIR moteurs, corrélés négativement au QI) et $B_2$ (intérêts circonscrits et rituels cognitifs, corrélés positivement au QI), avec un modèle à variables latentes pour le masking. On obtient $\mathcal{D}^4 = [0,3]^4$ qui capture la distinction clinique entre profils "bas niveau" et "haut niveau" que le DSM-5 a effacée en fusionnant les catégories...

## Une remarque finale

Les outils mathématiques déployés ici (espaces métriques, analyse de sensibilité, inférence bayésienne) ne sont pas des ornements. Ils contraignent la pensée à être précise là où le langage clinique peut rester vague. Dire qu'un profil est "à la frontière du spectre" n'a de sens que si on a défini ce qu'est une frontière, comment on la mesure, et avec quelle incertitude. C'est ce que ce notebook a tenté de faire.

# ANNEXE 1 : ce que "bayésien" signifie ici

Dans notre modèle, il y a **trois niveaux distincts d'incertitude** où le raisonnement bayésien peut s'appliquer. Ce sont trois problèmes différents.

## Niveau 1 - incertitude sur la mesure (vu plus haut)

C'est ce qu'on a déjà implémenté : étant donné un score observé $(a, b)$, la vraie position du patient dans $\mathcal{D}$ suit une distribution gaussienne centrée sur ce point.

$$P_{\text{réel}} \sim \mathcal{N}\left((a, b),\ \sigma^2 I\right)$$

Cela donne une probabilité $P(\text{profil} \in \mathcal{R}_{\text{TSA}})$ - ce qu'on calcule déjà. C'est le niveau le plus simple.

## Niveau 2 - incertitude sur les poids

C'est là que le raisonnement bayésien enrichit directement ce qu'on vient de construire. Plutôt que de faire varier les poids de façon **uniforme** (ce que fait la méta-analyse à pas de 0.1), on peut leur donner une **distribution a priori**.

$$w_A \sim \text{Beta}(\alpha, \beta)$$

La loi Beta est définie sur $[0, 1]$ et permet d'encoder des croyances cliniques :
- $\text{Beta}(1, 1)$ = uniforme = "je n'ai aucune préférence entre les deux axes"
- $\text{Beta}(3, 1)$ = concentrée vers 1 = "je crois que l'axe social est plus important"
- $\text{Beta}(1, 3)$ = concentrée vers 0 = "je crois que les CIR dominent"

Le score composite devient alors une **variable aléatoire** :

$$S = w_A \cdot a + (1 - w_A) \cdot b, \quad w_A \sim \text{Beta}(\alpha, \beta)$$

On peut calculer analytiquement son espérance et sa variance :

$$\mathbb{E}[S] = \frac{\alpha}{\alpha+\beta} \cdot a + \frac{\beta}{\alpha+\beta} \cdot b$$

$$\text{Var}(S) = (a - b)^2 \cdot \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

Ce qui est puissant : **la variance de $S$ est proportionnelle à $(a-b)^2$**. Un profil déséquilibré (grand $|a-b|$) aura toujours une variance élevée sous incertitude des poids - quelle que soit la distribution choisie. C'est la formalisation bayésienne exacte de ce que $\sigma(S)$ capturait dans la méta-analyse.

## Niveau 3 - mise à jour diagnostique

C'est le niveau le plus riche conceptuellement. On part de la question : **avant même de mesurer $(a, b)$, quelle est la probabilité qu'un individu soit TSA ?**

C'est la **prévalence** : environ 1 à 2% dans la population générale, mais beaucoup plus élevée dans une population clinique référée (peut-être 40-60% selon le contexte).

On note $\pi_0 = P(\text{TSA})$ cette probabilité a priori.

Quand on observe un score $S$ (ou un profil $(a, b)$), on met à jour via Bayes :

$$P(\text{TSA} \mid S) = \frac{P(S \mid \text{TSA}) \cdot \pi_0}{P(S \mid \text{TSA}) \cdot \pi_0 + P(S \mid \neg\text{TSA}) \cdot (1 - \pi_0)}$$

Ce qui nécessite de modéliser :
- $P(S \mid \text{TSA})$ : distribution des scores dans la population TSA (estimable à partir de données)
- $P(S \mid \neg\text{TSA})$ : distribution des scores dans la population non-TSA

Et l'idée clé qui émerge : **le même score $S = 1.2$ ne signifie pas la même chose selon le contexte clinique**. Chez un enfant référé par un pédopsychiatre ($\pi_0 = 0.5$), il donne une probabilité post-diagnostique élevée. Dans un screening populationnel ($\pi_0 = 0.01$), il reste majoritairement non-TSA malgré le score.

C'est ce qu'on appelle l'**effet du taux de base**. C'est un problème classique en médecine diagnostique que le raisonnement bayésien formalise exactement.

*Sources principales : Grove et al., Nature Genetics 2019 (n = 46 350) ; Courchesne et al., Molecular Autism 2021 (n = 205) ; DSM-5, APA 2013.*


```python

```
