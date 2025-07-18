---
title: "Entropie de Shannon"
date: '2025-06-01T12:11:55+02:00'
tags: ["Hasard", "Entropie", "Claude Shannon", "Théorie de l'information"]
categories: ["Sciences"]
cover:
  image: "/images/cover/shannon.jpg"   # Chemin vers ton image
  caption: "Claude Shannon" 
---

L’entropie de Shannon, c’est giga stylé !! J’espère vous plonger dans le même enthousiasme que le mien à la lecture de ceci.

# Introduction

Claude Shannon s’intéressait d’abord à la transmission d’information et a eu l’idée géniale d’introduire le concept d’entropie en informatique. Comme il est détaillé dans mon premier article sur une introduction générale au hasard, l’entropie avait déjà vu le jour en physique et avait déjà eu l’occasion d’engager de vifs débats, notamment entre Albert Einstein et Niels Bohr au congrès de Solvay de 1927.

Ici, le concept auquel répond Shannon avec son entropie est le suivant : comment mesurer la quantité d’information produite par une machine ? Claude Shannon a eu la merveilleuse idée de reformuler ceci par “combien de questions est-il nécessaire de poser afin de trouver l’élément suivant d’une chaine de données ?”.

Ce qui en découle est formidable.

De combien de bits un ordinateur a besoin pour générer le terme suivant dans une chaine de données ? Et ainsi, quel mot de passe est plus difficile à craquer : “Mdp1234!” ou “HF1$376b” ? Et oui car à chaque caractère de ces strings, dans une idée de bruteforce, on se demande quel sera le suivant.

# Formule

$$
\begin{align}
H(X) &= -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i) 
\text{où } \quad &p(x_i) = \text{probabilité de l'événement } x_i 
&n = \text{nombre d'événements possibles} 
&H(X) \text{ s'exprime en bits}
\end{align}
$$

# Exemples

## 1 lancer de pièce

### Cas théorique

Dans le cas théorique, pour le dire innocemment, c’est très peu prévisible. Il y a une grande place laissée au hasard.

Alors allons y, tout bêtement, combien de questions doit-on poser, au minimum, pour connaitre le résultat d’un lancer de pièce (pile/face) avec une probabilité égale de faire pile et de faire face ?

1 question : “est-ce pile ?”. Si oui, alors c’est pile et sinon c’est face. Vous voyez bien que c’est simple ! L’entropie est égale à 1 bit. En effet :

$$
H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5)) = 1 bit
$$

Dans ce cas, la théorie colle avec la pratique. En revanche, dans le cas d’une pièce truquée, il y a bien moins de place au “hasard”.

### Cas théorique d’une pièce truquée

Ici, c’est l’inverse ! Avec une pièce truquée, on est quelque peu certain du résultat qu’on obtiendra. D’autant plus que dans notre exemple ici nous allons choisir une pièce qui a une probabilité de faire pile égale à 0.9 et de faire ça égale à 0.1.

Alors, combien de questions ? D’après la formule de Shannon :

$$
H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1)) = 0.469 bit
$$

Il faut donc 0.469 question pour connaitre le résultat de la pièce truquée ! Il y a donc bien moins de hasard.

### Cas pratique d’une pièce truquée

En principe, l’entropie sur un unique lancer de pièce, truquée ou pas, doit être égale à 1 car on ne peut pas attribuer de fraction de bit dans l’encodage de Huffman. Pourtant il est bien optimal cet algorithme d’encodage de Huffman, mais surtout sur les blocs assez longs…

## 2 lancers youpiii

Regardons si notre encodage de Huffman permet de nous rapprocher de la valeur théorique de l’entropie de Shannon sur un lancer de pièce truquée.

Pour rappel, notre pièce a une probabilité de faire pile (P) de p=9/10 et de faire face (F) de 1/10.

D’abord, calculons quelques probabilités :

- Probabilité d’obtenir 2 fois pile : $$p² = 0.81$$
- Probabilité d’obtenir pile puis face = Probabilité d’obtenir face puis pile = $$p×(1-p) = 0.09$$
- Probabilité d’obtenir 2 fois face : $$(1-p)² = 0.01$$

Appliquons l’algorithme d’encodage de Huffman, afin d’optimiser le nombre de questions à poser pour savoir si c’est pile ou face, et établissons un graph comme suit et attribuer les arrêtes de façon à prioriser les évènements qui ont le plus de chance de se produire pour éviter de perdre du temps avec les évènements qui se produiront qu’une fois sur cent.

```plaintext
                 Racine (1.00)
                /             \
              0/               \1
              /                 \
          PP (0.81)         Nœud (0.19)
                          /           \
                        0/             \1  
                        /               \
                   FP (0.09)        Nœud (0.10)
                                   /         \
                                 0/           \1
                                 /             \
                            PF (0.09)      FF (0.01)
```

Ainsi, puisque le résultat sera PP dans 81% des situations, autant qu’il soit accessible via 1 bit (= 1 question) seulement et de la même façon, puisque FF se produira seulement dans 1% des cas, autant lui attribuer le plus gros nombre de bits (= le laisser en dernier recourt dans nos questions).

Nous pouvons donc établir l’encodage de l’information du résultat de la pièce suivant :

- PP = 1 bit
- FP = 2 bits
- PF = 3 bits
- FF = 3 bits

Et à présent, nous sommes en mesure de calculer l’entropie de Shannon dans cet exemple de deux lancers consécutifs d’une pièce truquée.

$$
0.81×1 + 0.09×2 + 0.09×3 + 0.01×3 = 1.29 bits par paire = 0.645 bit par lancer
$$

On obtient un résultat bien plus proche de la valeur théorique égale à 0.469 bit que de 1 bit !

Je vais me motiver à publier un notebook jupyter pour dessiner une jolie courbe et calculer les résultats pour n lancers.

---

*Dernière mise à jour le 01/06/2025*
