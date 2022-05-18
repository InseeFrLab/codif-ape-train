# Dictionnaires des variables présentes dans l'extraction de Théo

## I. Descriptions

Les données proviennent de 3 bases de données différentes et ont été fussionné grâce à l'identifiant de la liasse.

- Base de données ``SICORE`` qui contient les données issues de la classification faite par Sicore.
- Base de données ``LIASSE`` qui correspond aux données brute avant classification par Sicore.
- Base de données ``BILAN``.

La base ``BILAN`` est nécessaire pour contrôler la bonne mise en oeuvre de l'extraction des données mais n'est pas utile pour le modèle statistique. Pour ce dernier, seules les variables provenant de la base ``SICORE`` sont nécessaires ainsi que certaines variables non présentes dans ``SICORE`` mais seulement dans ``LIASSE``.

## II. Liste des variables

### 1. <u>Base de données ***Sicore***</u>

#### **• CFE**

Centre de Formalité d'Entreprise. Cette variable n'est plus présente dans Sirene 4 et ne doit donc pas être utilisée pour entrainer notre le statistique.

#### **• TYPE_SICORE**

Non identifiée, probablement à supprimer. Demander à Sirene s'il s'agit d'une variable pertinente ou non.

#### **• NAT_SICORE**

Nature de l'activité de l'entreprise.

| Modalité | Signification                                                       |
|----------|---------------------------------------------------------------------|
| 03       | Extraction                                                          |
| 04       | Fabrication, production                                             |
| 05       | Montage, installation                                               |
| 06       | Réparation                                                          |
| 07       | Transport                                                           |
| 08       | Import, export                                                      |
| 09       | Commerce de gros                                                    |
| 10       | Commerce de détail en magasin                                       |
| 11       | Profession libérale                                                 |
| 12       | Services                                                            |
| 13       | Location de meublés                                                 |
| 14       | Bâtiment, travaux publics                                           |
| 15       | Services aux entreprises                                            |
| 16       | Commerce de détail sur marchés                                      |
| 17       | Commerce de détail sur internet                                     |
| 20       | Location de logements                                               |
| 21       | Location de terrains et autres biens immobiliers                    |
| 22       | Promotion immobilière de bureaux                                    |
| 23       | Promotion immobilière de logements                                  |
| 24       | Promotion immobilière d’autres bâtiments                            |
| 25       | Réalisation de programmes de construction                           |
| 26       | Support de patrimoine familial immobilier sans activité de location |
| 99       | Autre                                                               |

#### **• APE_SICORE**

Code APE (Activité Principale Exercée) finalement retenu à la fin du processus de codification (soit grâce à Sicore, ou après intervention du gestionnaire).
Il existe 713 modalités différentes pour le code APE.

#### **• SED_SICORE**

Sédentarité de l'entreprise.

| Modalité   | Signification                                   |
|------------|-------------------------------------------------|
| A          | Ambulant                                        |
| E          | Ambulant hors France                            |
| F          | Forain                                          |


#### **• EVT_SICORE**

Evènement de la liasse, motif de sa création.
La dernière lettre de la modalité indique s'il s'agit d'une personne physique (P), morale (M), ou d'une exploitation en commun (F).

| Modalité   | Signification                                                      |
|------------|--------------------------------------------------------------------|
| 01*        | Création                                                           |
| 02*        | Création société sans activité                                     |
| 03*        | Création société avec activité hors siège                          |
| 04*        | Création suite 1er étab. d’entreprise étrangère                    |
| 05*        | Création par PP déjà enregistrée ou par texte                      |
| 07*        | Création entreprise étrangère sans étab.                           |
| 11*        | Transfert du siège de l’entreprise                                 |
| 12*        | Modification des principales activités de l’entreprise             |
| 24*        | Entrée de champ RCS, RM ou RSAC                                    |
| 51*        | Début d’activité au siège                                          |
| 52*        | Ouverture d’étab par entreprise sans activité                      |
| 53*        | Reprise d’un fond mis en location-gérance                          |
| 54*        | Ouverture d’un nouvel établissement                                |
| 55*        | Modification du nom de domaine du site internet d'un établissement |
| 60*        | Modification de l’identification de l’établissement                |
| 61*        | Adjonction d’activité                                              |
| 62*        | Suppression partielle d’activité                                   |
| 67*        | Modification des activités de l’établissement                      |


#### **• LIB_SICORE**

Libellé de la description d'activité de l'entreprise. Variable principale du modèle statistique.

#### **• DATE**

Date d'envoi de la liasse par l'entreprise.

### 2. <u>Base de données ***LIASSE***</u>

#### **• SURF**

Surface en $m^2$ de l'entreprise catégorisé en 5.

| Valeur | Signification            |
|--------|--------------------------|
| 1      | < 121 ou non renseigné   |
| 2      |  [121 ;400]              |
| 3      |  [401 ;2500]             |
| 4      | > 2500                   |


#### **• AUTO**

Type de la liasse.

| Valeur | Signification                         |
|--------|---------------------------------------|
| C      | Commerce                              |
| M      | Artisan                               |
| B      | Batellerie                            |
| L      | Pro.Libérale                          |
| R      | Agent commercial                      |
| A      | Agricole                              |
| G      | GIE-GEIE, soc.civ                     |
| S      | Asso                                  |
| I      | Impots                                |
| E      | Entr.Etrangere                        |
| D      | Origine RCS                           |
| N      | Origine RM                            |
| Y      | AE commerçant                         |
| Z      | AE artisan                            |
| X      | AE autre                              |



#### **• LIB_LIASSE_UL_U21**

Libellé de la description d'activité de l'unité légale. Normalement identique à ```LIB_LIASSE_ETAB_E71```

#### **• LIB_LIASSE_ETAB_E71**

Libellé de l'activité la plus importante de l'entreprise.

#### **• LIB_LIASSE_ETAB_E70**

Libellé de l'activité exercée par l'entreprise.

#### **• EVT_LIASSE**

Evènement de la liasse, motif de sa création.
La dernière lettre de la modalité indique s'il s'agit d'une personne physique (P), morale (M), ou d'une exploitation en commun (F).

| Modalité   | Signification                                                                                       |
|------------|-----------------------------------------------------------------------------------------------------|
| 01*        | Création                                                                                            |
| 02*        | Création société sans activité                                                                      |
| 03*        | Création société avec activité hors siège                                                           |
| 04*        | Création suite 1er étab. d’entreprise étrangère                                                     |
| 05*        | Création par PP déjà enregistrée ou par texte                                                       |
| 07*        | Création entreprise étrangère sans étab.                                                            |
| 10*        | Modification du nom ou du prénom de la personne                                                     |
| 11*        | Transfert du siège de l’entreprise                                                                  |
| 12*        | Modification des principales activités de l’entreprise                                              |
| 13*        | Modification de la forme juridique ou du statut particulier                                         |
| 14*        | Modification du ou des noms de domaine des sites internet                                           |
| 15*        | Modification du capital social                                                                      |
| 16*        | Modification de la durée de la personne morale ou de la date de clôture de l’exercice social        |
| 17*        | Modification de la mention « associé unique » (déclaration ou suppression)                          |
| 18*        | Economie Sociale et Solidaire (ESS)                                                                 |
| 19*        | Changement de la nature de la gérance                                                               |
| 20*        | Modification de la date de début d‘activité                                                         |
| 21*        | Reprise d’activité de l’entreprise après une cessation temporaire                                   |
| 22*        | Dissolution                                                                                         |
| 23*        | Demande de renouvellement du maintien provisoire de l’immatriculation au RCS                        |
| 24*        | Entrée de champ RCS, RM ou RSAC                                                                     |
| 25*        | Déclaration, modification relative à l’EIRL                                                         |
| 26*        | Reconstitution des capitaux propres                                                                 |
| 27*        | Sortie de champ du Répertoire des Métiers                                                           |
| 28*        | Dissolution suite à décision de l'associé unique Personne Morale                                    |
| 29*        | Autre modification concernant la personne morale                                                    |
| 30*        | Modification relative aux membres d’un groupement                                                   |
| 31*        |                                                                                                     |
| 32*        | Modification relative aux associés non gérants relevant du régime TNS - MSA                         |
| 33*        | Modification relative aux dirigeants d’un groupement                                                |
| 34*        | Modification relative aux dirigeants d’une société de personnes                                     |
| 35*        | Modification relative aux dirigeants d’une SARL ou d’une société de capitaux                        |
| 36*        | Modification relative au représentant social d’une Soc. Etr. employeur sans établissement en France |
| 40*        | Cessation temporaire d’activité de l’entreprise                                                     |
| 41*        | Cessation totale d’activité non salariée                                                            |
| 42*        | Décès de l’exploitant individuel sans poursuite de l’exploitation                                   |
| 43*        | Cessation totale d’activité avec demande de maintien provisoire au RCS ou au RM                     |
| 44*        | Cessation d’activité agricole                                                                       |
| 45*        | Cessation d’activité agricole avec conservation de stocks ou de cheptel                             |
| 46*        | Départ en retraite avec conservation d’une exploitation de subsistance                              |
| 47*        | Option TVA bailleur de biens ruraux                                                                 |
| 51*        | Début d’activité au siège                                                                           |
| 52*        | Ouverture d’étab par entreprise sans activité                                                       |
| 53*        | Reprise d’un fond mis en location-gérance                                                           |
| 54*        | Ouverture d’un nouvel établissement                                                                 |
| 55*        | Modification du nom de domaine du site internet d'un établissement                                  |
| 56*        | Transfert d’un établissement                                                                        |
| 60*        | Modification de l’identification de l’établissement                                                 |
| 61*        | Adjonction d’activité                                                                               |
| 62*        | Suppression partielle d’activité                                                                    |
| 63*        | Acquisition du fonds par l’exploitant                                                               |
| 64*        | Renouvellement du contrat de location gérance                                                       |
| 65*        | Embauche d’un premier salarié dans un établissement                                                 |
| 66*        | Fin d’emploi de tout salarié dans un établissement                                                  |
| 67*        | Modification des activités de l’établissement                                                       |
| 68*        | Changement de locataire - gérant                                                                    |
| 70*        | Modification relative à une personne ayant le pouvoir d’engager l’établissement                     |
| 80*        | Fermeture d’un établissement                                                                        |
| 81*        | Fin d’activité au siège, qui reste siège                                                            |
| 82*        | Mise en location gérance ou en gérance mandat d’un des fonds exploités                              |
| 83*        | Mise en location gérance du fonds unique sans maintien au RCS ou au RM                              |
| 84*        | Mise en location gérance du fonds unique ou en gérance mandat avec maintien au RCS                  |
| 90*        | Radiation d’office du RSI                                                                           |
| 91*        | Rejet de l’immatriculation au RM                                                                    |
| 92*        | Rejet de la demande d’inscription au RCS                                                            |
| 93*        | Radiation d’office du RM                                                                            |
| 94*        | Radiation d’office du RCS                                                                           |
| 95M        | Invalidation de la mention Economie Sociale et Solidaire (ESS)                                      |
| 95P        | Demande d’inscription d’un gérant majoritaire au répertoire SIRENE                                  |
| 96*        | Réactivation suite à radiation d’office                                                             |
| 97*        | Sortie du Micro Social (MSS) par franchissement de seuils ou par option                             |
| 98*        | Refus d’immatriculation au RCS                                                                      |
| 99*        | Correction ou complément d’une formalité                                                            |

#### **• SED_LIASSE**

Sédentarité de l'entreprise.

| Modalité   | Signification                                   |
|------------|-------------------------------------------------|
| A          | Ambulant                                        |
| E          | Ambulant hors France                            |
| F          | Forain                                          |

### 3. <u>Base de données ***BILAN***</u>

#### **• TYPE_BILAN**

Non identifiée, probablement à supprimer. Demander à Sirene s'il s'agit d'une variable pertinente ou non.

#### **• APE_BILAN**

Code APE. Utilisé pour vérifier la qualité de l'extraction.

#### **• TROUVE_BILAN**

Nécessaire pour vérifier l'extraction.

#### **• TROUVE_XML**

Nécessaire pour vérifier l'extraction.
