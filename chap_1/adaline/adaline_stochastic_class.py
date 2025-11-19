import numpy as np

class AdalineSGD:
    """Classificador AdalineGD (Adaptive Linear Neuron) treinado por gradiente descendente em batch.

    Parâmetros
    ----------
    eta : float
        Taxa de aprendizagem (tipicamente entre 0.0 e 1.0).
    n_iter : int
        Número de épocas (iterações) de treinamento.
    shuffle : bool, default=True
        Se True, os dados serão embaralhados a cada época para evitar ciclos.
    random_state : int
        Semente do gerador aleatório usada para inicializar os pesos.

    Atributos
    ----------
    w_ : ndarray, shape = [n_features]
        Vetor de pesos ajustados após o treinamento.
    b_ : float
        Termo de bias ajustado após o treinamento.
    losses_ : list
        Valores da função de perda (mean squared error) registrados por época.

    Observações
    ----------
    A função de ativação é linear (identidade). O ajuste é feito por gradiente descendente em batch,
    minimizando o erro quadrático médio.
    """

    # define os parametros a serem iniciados pela classe
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
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

        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                loss = self._update_weights(xi, target)
                losses.append(loss)
        return self

    def partial_fit(self, X: np.array, y:np.array) -> 'Perceptron':
        """ Ajusta os dados sem reinicializar os pesos """
        if not self.w_initialized:
            # supoe que nao foi inicializado os pesos
            self._initialize_weights(X.shape[1])
        # ravel achata o array
        # fornece o numero de elementos no array, idependente das dimensoes e formato
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X: np.array, y: np.array ,) -> tuple[np.array, np.array]:
        """ Embaralha os dados de treinamento """
        r = self.rgen_.permutation(len(y))
        return X[r], y[r]

    # ja era feito na classe Perceptron, mas agora virou uma funcao separada
    def _initialize_weights(self, m: int) -> None:
        """ Inicializa os pesos com zeros """
        self.rgen_ = np.random.RandomState(self.random_state)
        self.w_ = self.rgen_.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = float(0.)
        self.w_initialized = True

    def _update_weights(self, xi: np.array, target: float) -> float:
        """ Atualiza os pesos para um unico exemplo de treinamento """
        # calcula a saida da rede
        output = self.net_input(xi)
        # calcula o erro
        error = (target - output)
        # atualiza os pesos
        self.w_ += self.eta * xi.dot(error)
        self.b_ += self.eta * error
        # calcula o erro quadratico medio
        loss = 0.5 * (error**2)
        return loss
        
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