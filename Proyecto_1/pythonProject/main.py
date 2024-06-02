import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

datos = pd.read_csv("smogon.csv")


#TDF-IDF

#para generar la matriz tf-idf utilizando bigramas:

#utilizamos tdfifvectorizer para añadir una lista de stopwords en ingles de scikit-learn
columna_bigramas= datos['moves']
vec = TfidfVectorizer(ngram_range=(2,2), stop_words="english")

#para generar la matriz tf-idf
generar_matriz = vec.fit_transform(columna_bigramas)
#definir numero total de tokens
tokens= vec.get_feature_names_out()
num_tokens =len(tokens)

#crear dataframe con la matriz tf-idf
tfidf_df= pd.DataFrame(data=generar_matriz.toarray(), columns=tokens)

print("Número total de tokens:", num_tokens)
print("Ejemplo de tokens:", tokens[:10])
print("Nueva matriz TF-IDF:")
print(tfidf_df.head())

#para agrupar con k-medias
km = KMeans(n_clusters=4, n_init=40)
tfidf_df['cluster'] = km.fit_predict(tfidf_df)

#guarda dataframe con clusters
tfidf_df.to_csv("smogon_cluster.csv", index=False)

print("\nEl cluster asignado a cada documento es:")
print(tfidf_df['cluster'].value_counts())


