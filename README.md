# cam137.org

Blog personnel propulse par [Hugo](https://gohugo.io/), heberge sur [cam137.org](https://cam137.org).

## Prerequis

- [Hugo](https://gohugo.io/installation/) (version extended recommandee)
- Git

## Developpement local

Cloner le depot puis lancer le serveur de developpement :

```bash
git clone https://github.com/guepardlover77/cam137.org.git
cd cam137.org
hugo server -D
```

Le site est accessible sur `http://localhost:1313`.

## Nouveau post

```bash
hugo new posts/mon-article.md
```

Editer le fichier genere dans `content/posts/`, puis previsualiser avec `hugo server -D`.

## Deploiement

Le site est deploye automatiquement via GitHub Actions a chaque push sur la branche principale.
