---
title: "Les théorèmes d'incomplétude de Gödel 1/ - Définitions et Peano"
date: '2025-06-11T11:58:30+02:00'
tags: ["Kurt Gödel", "Logique", "Maths"]
categories: ["Sciences"]
cover:
  image: "images/cover/godel.jpg"
  caption: "Kurt Gödel"
---

## Système formel
Un système formel est une construction mathématique rigoureuse qui vise à capturer et organiser notre raisonnement logique. Il se compose de trois éléments essentiels :
- Les symboles de base forment l'alphabet du système. <br>
Dans un système arithmétique, on trouve par exemple les chiffres 0, 1, 2..., les opérateurs +, ×, les symboles logiques comme ∧ (et), ∨ (ou), ¬ (non), ainsi que des variables et des quantificateurs ∀ (pour tout) et ∃ (il existe). <br>
- Les règles de formation déterminent comment construire des formules bien formées à partir de ces symboles. <br>
Elles précisent quelles combinaisons de symboles ont un sens mathématique. Par exemple, "∀x (x + 0 = x)" est une formule bien formée, tandis que ")0 + ∀x(" ne l'est pas.
- Les règles d'inférence constituent le moteur du raisonnement. <br>
Elles spécifient comment dériver de nouvelles formules à partir de formules déjà établies. Le modus ponens en est l'exemple le plus classique : si nous avons "A implique B" et "A", alors nous pouvons conclure "B".
À partir de ces éléments, un système formel distingue deux types de formules : les axiomes, qui sont des vérités admises sans démonstration, et les théorèmes, qui sont des formules dérivées des axiomes par application répétée des règles d'inférence.

## Définitions clés

### Complétude et incomplétude
Un système formel est dit complet si toute formule bien formée du système est soit démontrable (c'est un théorème), soit réfutable (sa négation est un théorème). <br>
En d'autres termes, pour toute proposition P du système, soit P soit ¬P est démontrable. <br>
Cette propriété semble naturelle : intuitivement, toute affirmation mathématique devrait être soit vraie soit fausse, et un système parfait devrait pouvoir le déterminer. <br>
L'incomplétude désigne précisément l'absence de cette propriété : il existe des formules qui ne peuvent être ni démontrées ni réfutées dans le système.

### Consistance et inconsistance
La consistance (ou cohérence) d'un système signifie qu'il ne peut pas démontrer à la fois une formule et sa négation. <br>
Un système consistant ne peut pas prouver que "2 + 2 = 4" et "2 + 2 ≠ 4" simultanément. Cette propriété est cruciale car un système inconsistant permet de démontrer n'importe quoi, ce qui le rend mathématiquement inutile. <br>
L'inconsistance est donc l'existence d'au moins une formule P telle que le système peut démontrer à la fois P et ¬P. <br>
Cette situation est catastrophique pour un système formel car elle entraîne ce qu'on appelle le "principe d'explosion" : de une contradiction, tout peut être déduit.

### Décidabilité
Un système est décidable s'il existe une procédure mécanique (un algorithme) qui, pour toute formule du système, détermine en un nombre fini d'étapes si cette formule est un théorème ou non. En gros, si on arrive à se décider. <br>


## L'arithmétique de Peano comme exemple

L'arithmétique de Peano constitue l'un des exemples les plus importants de système formel. <br>
Développée par Giuseppe Peano à la fin du XIXe siècle, elle vise à formaliser nos connaissances sur les nombres entiers naturels et leurs opérations.

### Les axiomes de Peano
- Zéro est un nombre naturel : 0 ∈ ℕ
- Tout nombre naturel a un successeur : ∀n ∈ ℕ, ∃s(n) ∈ ℕ
- Zéro n'est le successeur d'aucun nombre : ∀n ∈ ℕ, s(n) ≠ 0
- Des nombres différents ont des successeurs différents : ∀n,m ∈ ℕ, s(n) = s(m) → n = m
- Principe de récurrence : Si une propriété est vraie pour 0 et que, chaque fois qu'elle est vraie pour un nombre n, elle l'est aussi pour s(n), alors elle est vraie pour tous les nombres naturels.

### La conjecture de Goodstein
D'abord, on prend un nombre. Ensuite, on l'écrit en base 2. Puis on remplace les 2 par des 3. Et enfin on soustrait 1. <br>
M.Goodstein nous dit que cette séquence commençant par n tend vers 0 pour tout n. <br>
Cet exemple est troublant parce qu'elle a été démontrée vraie en 1982 mais sa démonstration a nécessité l'utilisation de l'induction transfinie sur des ordinaux bien au-delà de ce que peut exprimer l'arithmétique de Peano. <br>
Rien que ça.

---

L'arithmétique de Peano peut exprimer des concepts mathématiques très sophistiqués.
Elle peut définir l'addition, la multiplication, les relations d'ordre, et même encoder des notions complexes comme la primalité d'un nombre ou la conjecture de Goldbach. <br>
Cette richesse expressive est précisément ce qui rend l'arithmétique de Peano vulnérable aux théorèmes de Gödel. Sa capacité à "parler d'elle-même" - propriété que nous explorerons avec la numérotation de Gödel - est à la fois sa force et sa faiblesse. <br>

*Dernière mise à jour le 11/07/2025*

