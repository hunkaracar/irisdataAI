#iris veri setinin yüklenmesi

from turtle import shape
import numpy as np
from sklearn.datasets import load_iris #yapay öğrenme alanında kullanılan bir kütüphane
iris_dataset = load_iris()                                                                    #hedefler: setosa,Versicolor,Virginia
#print(iris_dataset)                                                                          #hedefler(numara):0,1,2

#verinin ikiye bölünmesi(makinenin öğrenmesi ve veriyi test etmesi)

from sklearn.model_selection import train_test_split
X_ogren, X_test , y_ogren , y_test = train_test_split(iris_dataset["data"],iris_dataset["target"])
print(X_ogren,shape)
print(X_test,shape)

#uygun modeli seçme

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)

#öğrenme

knn1 = knn.fit(X_ogren,y_ogren)
print(knn1)

#tahmin

X_yeni = [[3.5,2.1,3.4,1.2]]    #yeni değerler, makineye verdiğimiz değerleri kullanarak(veri seti) tahmin yapması
tahmin = knn.predict(X_yeni)
print(tahmin)

#doğruluk ve test verisi

dogruluk = knn.predict(X_test)
print(dogruluk)

truee = np.mean(dogruluk == y_test)*100 #dogruluk yüzdesi
print("makine öğrenmesi doğruluk yüzdesi:{0}".format(truee))


