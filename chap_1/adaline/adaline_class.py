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
        self.b_ = float(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors)/X.shape[0]
            self.b_ += self.eta*2.0*errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X: np.array) -> np.ndarray:
        """ Calcula o valor total da rede"""
        # z = w^t*X + b
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X: np.array) -> np.ndarray:
        """ Função de ativação linear """
        # essa função é identidade
        return X

    def predict(self, X: np.array) -> np.ndarray:
        """ Simula a função delta """
        return np.where(self.activation(self.net_input(X) >= 0.5, 1, 0))