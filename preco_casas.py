import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score

df_casas = pd.read_csv('/content/preco_casas.csv')

df_casas.head()

df_casas.isnull().sum()

X=df_casas.drop(['date','price','city','street','statezip','country'],axis=1)
y=df_casas['price']

X_treino, X_teste, y_treino, y_teste=train_test_split(X,y,test_size=0.3, random_state=0)

"""## regressão linear"""

regressao = LinearRegression()
regressao.fit(X_treino,y_treino)
previsao = regressao.predict(X_teste)

r2_score(y_teste,previsao)

"""## arvore de decisao"""

arvore = DecisionTreeRegressor()
arvore.fit(X_treino,y_treino)
previsao = arvore.predict(X_teste)

r2_score(y_teste,previsao)

"""## random forest"""

random = RandomForestRegressor()
random.fit(X_treino,y_treino)
previsao = random.predict(X_teste)

r2_score(y_teste,previsao)

"""## redes neurais"""

redes = MLPRegressor()
redes.fit(X_treino,y_treino)
previsao = redes.predict(X_teste)

r2_score(y_teste,previsao)