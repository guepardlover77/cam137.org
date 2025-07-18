---
title: "Compagnies de bus"
date: '2025-07-11T10:08:30+02:00'
tags: ["Informatique", "Aventure"]
categories: ["Sciences"]
---

Je cherchais un moyen de récupérer tous les mails des sociétés de transports de la région Nouvelle-Aquitaine. <br>
Pour ça je suis allé sur le site de la Fédération National des Transports de Voyageurs qui propose un annuaire de leurs adhérents. <br>
Ensuite j'ai remarqué qu'on pouvait filtrer et, parmi les résultats, cliquer sur "voir la fiche" pour afficher une petite fenêtre avec les informations de la compagnie. <br>
Le problème c'est que l'URL ne change pas. Même pas quand on applique des filtres. Vous sentez la douille ? <br>
Naturellement, les boutons "Voir la fiche" n'apparaissent pas dans le code source de la page. Tout se faisait par script JS. <br>
Lorsque le bouton est cliqué, un fichier JS envoie une requête pour récupérer un fichier JSON avec toutes les infos et afficher tout ça correctement. <br>
Le code JS est écrit sur une seule ligne et fait 76848 caractères donc autant vous dire que j'irai pas lire le truc. <br>

## Méthode
D'abord j'ai regardé les requêtes et j'ai compris quel fichier se compilait en cliquant sur des boutons bien précis. <br>
Ensuite il suffisait de récupérer la réponse de la requête qui est ni plus ni moins qu'un JSON avec TOUTES les infos lol. <br>
Le JSON se présente comme ça :
```
{'total': int,
 'totalPages': int,
 'data': [{'id': int,
   'name': str,
   'type': str,
   'sub_type': str,
   'regional_union': str,
   'phone': str,
   'email': str,
   'zipCode': str,
   'address': str,
   'city': str,
   'latitude': float,
   'longitude': float,
   'helper_days': None,
   'helper_time': None}, 
		{...}]}
```

Et après ce que je veux c'est le nom de la compagnie ainsi que le mail pour pouvoir copier coller ça dans un excel et envoyer un mail à tout le monde en même temps. <br>
Pour cela, il suffit d'un script python :

```
import pandas as pd
import json

with open('blabla.json') as f:
    data = json.load(f)['data']

df = pd.DataFrame([[d['name'], d['email']] for d in data], columns=['Nom', 'Mail'])
df.to_excel("output.xlsx", index=False)
```

## Pourquoi fonctionner comme ça ?

Déjà c'est plus léger parce que le backend envoie que des données et pas de HTML entier mais en réalité est-ce que ça change grand chose ?
Dans tous les cas on dépassera pas Netflix en bande passante. <br>

L'idée de traiter de la donnée c'est que c'est beaucoup plus modulable et facile à adapter pour un appareil mobile ou une appli.
