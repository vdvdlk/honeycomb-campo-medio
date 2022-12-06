import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import pi
from tqdm.auto import tqdm, trange

# graficos = True
graficos = False

CI = complex(0, 1)

Lx = 100
Ly = 100
N = Lx * Ly  # Número de células unitárias

mu = 0.0  # Potencial químico
beta = 16.0  # 1 / (k_B * T)

# Parâmetros de hopping
t1 = -1.0
t2 = -1.0
t3 = -1.0
t4 = -1.0
t5 = -1.0
t6 = -1.0


def array_k(Lx: int = Lx, Ly: int = Ly) -> np.ndarray:

    if Lx % 2 == 0:
        n_x_inicial = - (Lx / 2 - 1)
        n_x_final = Lx / 2
    else:
        n_x_inicial = - (Lx - 1) / 2
        n_x_final = - n_x_inicial

    array_n_x = np.arange(
        start=n_x_inicial,
        stop=n_x_final + 1
    )
    array_k_x = 2 * pi * array_n_x / Lx

    if Ly % 2 == 0:
        n_y_inicial = - (Ly / 2 - 1)
        n_y_final = Ly / 2
    else:
        n_y_inicial = - (Ly - 1) / 2
        n_y_final = - n_y_inicial

    array_n_y = np.arange(
        start=n_y_inicial,
        stop=n_y_final + 1
    )
    array_k_y = 2 * pi * array_n_y / Ly

    return array_k_x, array_k_y


array_k_x, array_k_y = array_k()


def hamiltoniana(k_x: float, k_y: float, mu: float = mu, t1: float = t1, t2: float = t2, t3: float = t3, t4: float = t4, t5: float = t5, t6: float = t6) -> np.ndarray:

    H = np.zeros((10, 10), dtype=complex)

    H[0, 1] = t1
    H[0, 3] = t6 * np.exp(- CI * k_x)
    H[0, 9] = t5 * np.exp(- CI * k_y)
    H[1, 2] = t2
    H[1, 4] = t4
    H[2, 3] = t3
    H[4, 5] = t5
    H[5, 6] = t6
    H[5, 8] = t1
    H[6, 7] = t3
    H[7, 8] = t2 * np.exp(CI * k_x)
    H[8, 9] = t4

    H[1, 0] = t1
    H[3, 0] = t6 * np.exp(CI * k_x)
    H[9, 0] = t5 * np.exp(CI * k_y)
    H[2, 1] = t2
    H[4, 1] = t4
    H[3, 2] = t3
    H[5, 4] = t5
    H[6, 5] = t6
    H[8, 5] = t1
    H[7, 6] = t3
    H[8, 7] = t2 * np.exp(- CI * k_x)
    H[9, 8] = t4

    # H += - mu * np.identity(n=10, dtype=complex)

    return H


def dispersao(array_k_x: np.ndarray = array_k_x, array_k_y: np.ndarray = array_k_y):

    bandas = np.zeros(shape=(array_k_x.size, array_k_y.size, 10))
    autovetores = np.zeros(
        shape=(array_k_x.size, array_k_y.size, 10, 10),
        dtype=complex
    )

    i = 0
    for k_x in tqdm(array_k_x, desc='Relação de dispersão'):
        j = 0
        for k_y in array_k_y:
            H = hamiltoniana(
                k_x=k_x,
                k_y=k_y
            )
            w, v = np.linalg.eigh(
                a=H,
                UPLO='U'
            )
            bandas[i, j, :] = w  # índices: kx, ky, banda
            autovetores[i, j, :, :] = v  # índices: kx, ky, componente, banda
            j += 1
        i += 1

    return bandas, autovetores


bandas, autovetores = dispersao()
banda_5, autovetores_5 = bandas[:, :, 4], autovetores[:, :, :, 4]
banda_6, autovetores_6 = bandas[:, :, 5], autovetores[:, :, :, 5]


def f_FD(E: np.ndarray, mu: float = mu, beta: float = beta) -> float:
    exponencial = np.exp(beta * (E - mu))
    return 1 / (1 + exponencial)


def densidade(bandas: np.ndarray = bandas, N: int = N, mu: float = mu, beta: float = beta) -> float:
    return 2 * np.sum(f_FD(bandas, mu, beta)) / (N * 10)


def cinetica(bandas: np.ndarray = bandas, N: int = N, mu: float = mu, beta: float = beta) -> float:
    return 2 * np.sum(bandas * f_FD(bandas, mu, beta)) / (N * 10)


def valor_esp_cjcj(epsilon: np.array, U: np.array, mu: float = mu, beta: float = beta) -> np.ndarray:

    array_1 = np.abs(U) ** 2
    array_2 = f_FD(
        E=epsilon,
        mu=mu,
        beta=beta
    )

    return np.matmul(array_1, array_2)


def valor_esp_n(bandas: np.ndarray = bandas, autovetores: np.ndarray = autovetores):

    n_i = np.zeros(
        shape=bandas.shape[2],
        dtype=float
    )

    for i in trange(bandas.shape[0], desc='n(i)'):
        for j in range(bandas.shape[1]):
            n_i += 2 * valor_esp_cjcj(
                epsilon=bandas[i, j, :],
                U=autovetores[i, j, :, :]
            )

    return n_i / (bandas.shape[0] * bandas.shape[1])


density = densidade()
print('density =', density)

kinectic = cinetica()
print('kinectic =', kinectic)

n_i = valor_esp_n()
print('n(i) ->', n_i)


if graficos == True:

    x, y = np.meshgrid(
        array_k_x,
        array_k_y,
        indexing='ij'
    )

    fig_fermi, ax_fermi = plt.subplots()
    ax_fermi.set_xlabel('$k_x$')
    ax_fermi.set_ylabel('$k_y$')
    ax_fermi.contour(
        x,
        y,
        banda_6
    )

    fig_bandas, ax_bandas = plt.subplots(
        subplot_kw={"projection": "3d"}
    )
    ax_bandas.set_xlabel('$k_x$')
    ax_bandas.set_ylabel('$k_y$')
    for j in range(10):
        ax_bandas.plot_surface(
            x,
            y,
            bandas[:, :, j]
        )

    fig_dispersion, ax_dispersion = plt.subplots(
        subplot_kw={"projection": "3d"}
    )
    ax_dispersion.set_xlabel('$k_x$')
    ax_dispersion.set_ylabel('$k_y$')
    ax_dispersion.plot_surface(
        x,
        y,
        banda_5,
        cmap=cm.Blues_r
    )
    ax_dispersion.plot_surface(
        x,
        y,
        banda_6,
        cmap=cm.Reds
    )

    plt.show()
