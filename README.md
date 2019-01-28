# handwritten_char_recognition
recognition of handwritten characters using a neural network

Auteur : Nicolas Morlier

prérequis :
-python 3.6 ou supérieur (mais inférieur à 3.7)
-numpy, pandas, sklearn, tensorflow, tensorboard, opencv-python, keras, matplotlib, ipython
1) Télécharger le programme :
git clone https ://github.com/Themousaillon/handwritten_char_recognition.git
2) Lancer le programme : ipython -i main.py
3) Charger une image : im = cv2.imread(path, 0) où path est le chemin vers l’image
4) prédiction : predictTools.predict_lettre(model, im)
5) graphe des kernels : plotter.plotKernels(model, 4)
6) graph des convolutions avec les kernels : plotter.plotConvolve(im, 4, model)
7) Lancer l’interface graphique tensorboard : tensorboard --logdir Graph/
! ! les chemins sont à écrire entre guillemets ex : "mondossier/lettreA.png" ! !
! ! Il faut être à la racine du projet pour lancer les commandes ! !
