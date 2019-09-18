# USAGE
# python trabFinal.py --training images/training --testing images/testing

# import the necessary packages
from lbp.localbinarypatterns import LocalBinaryPatterns
from functions import functions
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True)
ap.add_argument("-e", "--testing", required=True)
args = vars(ap.parse_args())
 
#inicializa nomdes dos classificadores
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

#inicializa classificadores
classifiers = [
    KNeighborsClassifier(3),
    LinearSVC(C=100.0, random_state=42),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]	

#Obtem dados para o treinamento (vetor de características + label)
(X_train, Y_train) = functions.Training(args)

# treina o classificador com os dados obtidos					
for model in classifiers:	
	model.fit(X_train, Y_train)

#Obtém dados para o teste (vetor de características)
allData = functions.Testing(args)

#Faz a predição para cada classificador
results = functions.Prediction(classifiers,names,allData)

#Imprime resultados da predição
functions.PrintResults(results)