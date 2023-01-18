# Recoinize

An AI based project to recognize euro coins.

## Possible implementation

### Entrainement

On entraine le réseau (en mode siamois, ou triplet) à trouver les meilleurs paramètres pour qu'à chaque image,
il associe un vecteur de sortie le plus dicriminant possible.

-- l'entrainement s'arrête ici --

Une fois entièrement entrainé, on utilise ce réseau en mode seul.
On lui passe toutes les images du dataset (~ 100 000).

On peut nous même faire une ACP pour trouver les 3 composantes principales, et afficher tous les points en 3D
(ou 2 composantes en 2D).

### Validation

#### Mode 1

On envoie l'image que l'on cherche à classifier dans notre réseau (en mode seul), ce qui nous donne un vecteur unique à l'image permettant de l'identifier.

On labélise cette pièce par la classe dominante des k plus proches voisins sur les 100 000 images (algorithme de simplification de Stuff Made Here
pour éviter de tester tous les cas)

#### Mode 2

On envoie l'image que l'on cherche à classifier dans notre réseau (en mode seul), ce qui nous donne un vecteur unique à l'image permettant de l'identifier.

Pour chaque classe, on calcule la distance entre ce vecteur et le vecteur de la classe.
On obtient alors 276 pourcentages, on prend le plus élevé.

#### Mode 3 (possiblement inutile, à supprimer)

Pour chaque classe (276), on prend l'image à tester, et une image représentant la classe. 