from lbp.localbinarypatterns import LocalBinaryPatterns
import cv2
from imutils import paths
import os

def Prediction(classifiers, names, allData):	
	results = []
	for (index,model) in enumerate(classifiers):
		hits = {"angry":0,"happy":0}
		total = {"angry":0,"happy":0}
		for hist in allData:
			prediction = model.predict(hist[0].reshape(1, -1))
		
			label = hist[1]
			if(label == "angry"):
				total["angry"] += 1
				if(label == prediction[0]):
					hits["angry"] += 1
			elif(label == "happy"):
				total["happy"] += 1
				if(label == prediction[0]):
					hits["happy"] += 1
		results.append((names[index],hits,total))
	
	return results

def PrintResults(results):
	for result in results:						
		print("\nResultado", result[0])
		if(result[2]["angry"] !=0 ):
			print("Angry Hits",  result[1]["angry"],end ='|') 
			print("Total", result[2]["angry"],end ='|') 
			print("Score", result[1]["angry"]/result[2]["angry"]) 
		
		if(result[2]["happy"] !=0 ):
			print("Happy Hits",  result[1]["happy"],end ='|') 
			print("Total", result[2]["happy"],end ='|') 
			print("Score", result[1]["happy"]/result[2]["happy"]) 
	
def Training(args):	
	X_train = []
	Y_train = []
	for imagePath in paths.list_images(args["training"]):	
		data = ViolaJonesWithLBP(imagePath)
		X_train.append(data[0])
		Y_train.append(data[1])
	
	return (X_train,Y_train)
	
def Testing(args):
	allData = []
	#Percorre as imagens utilizadas no teste para obter o vetor de características
	for imagePath in paths.list_images(args["testing"]):
		data = ViolaJonesWithLBP(imagePath)
		allData.append(data)
	
	return allData
	
def ViolaJonesWithLBP(imagePath):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)	
	desc = LocalBinaryPatterns(24, 8)
	
	#percorre as faces encontradas
	for (x,y,w,h) in faces:
		face = image[y:y+h, x:x+w]
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)			
		histogram = [0]
		
		#Obtém o vetor de características de cada parte do rosto (olhos, nariz e boca) e unifica em um vetor
		for (ex,ey,ew,eh) in eyes:
			gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(gray)
			histogram = histogram + hist
		
		if(len(histogram) <= 1):
			continue
		else:
			return (histogram,imagePath.split(os.path.sep)[-2])
			
	return (-1,-1)