import numpy as np

class Perceptron:
    """ Classficador Perceptron

    Parametros
    ----------
    eta : float
        Taxa de aprendizagem - [0,1]
    n_iter : int
        Itera na base de dados de treinamento (epochs)
    random_state : int
        Gerador de semente numeros aleatorios para inicializacao dos pesos (w)
    

    Atributos
    ----------
    w_ : 1d-vetor
        pesos apos serem 'fitados'
    b_ : escalar
        termo de interseccao da reta depois de ser 'fitado' (bias)
    
    errors_ : list
        numero de classificacoes erradas (updates) em cada iteração (epoch)
    
    """

    # define os parametros a serem iniciados pela classe
    def __init__(self, eta=0.01, n_iter=50, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    # define a funcao fit para o dataset de treinamento
    def fit(self, X: np.array, y:np.array) -> 'Perceptron':
        """ ajustar os dados para o treinamento.

        Parametros:
        -----------
        # note como podem ser mais de um vetor pela chave {}
        X : {tipo-vetor}, shape = [n_examples, n_features]
            Vetores de treinamento, onde n_examples eh o numero de examples e 
            n_features eh o numero de features
        y : tipo-vetor, shape = [n_examples]
            valores alvo

        Returns 
        -------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        # pesos levemente diferentes de zero
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            # xi eh i-esimo elemento de X
            for xi, target in zip(X, y):
                updates = self.eta * (target - self.predict(xi))
                # vetor multiplicando outro vetor
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        return self

    def net_input(self, X: np.array) -> np.ndarray:
        """ Calcula o valor total da rede"""
        # z = w^t*X + b
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X: np.array) -> np.ndarray:
        """ Simula a função delta """
        return np.where(self.net_input(X) >= 0.0, 1, 0)