import cv2
import numpy as np
import matplotlib.pyplot as plt

def converter_para_preto_branco(imagem, limiar=128):
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplica a limiarização para converter para preto e branco
    _, imagem_pb = cv2.threshold(imagem_cinza, limiar, 255, cv2.THRESH_BINARY)

    return imagem_pb

def reduzir_niveis_de_cinza(imagem, niveis=16):
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Normaliza a imagem para ter valores entre 0 e 1
    imagem_normalizada = imagem_cinza / 255.0

    # Reduz os níveis de cinza
    imagem_reduzida = np.floor(imagem_normalizada * niveis) * (255 / (niveis - 1))

    # Converte a imagem de volta para o formato original
    imagem_reduzida = imagem_reduzida.astype(np.uint8)

    return imagem_reduzida

# Carrega a imagem colorida
imagem = cv2.imread('naruto.png')

# Reduz os níveis de cinza (exemplo: 16 níveis)
imagem_reduzida = reduzir_niveis_de_cinza(imagem, niveis=16)

# Converte a imagem para preto e branco
imagem_preto_branco = converter_para_preto_branco(imagem, limiar=128)

# Exibe a imagem original, a imagem com níveis de cinza reduzidos e a imagem em preto e branco
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
plt.title('Imagem Colorida Original')

plt.subplot(1, 3, 2)
plt.imshow(imagem_reduzida, cmap='gray')
plt.title('Imagem com Níveis de Cinza Reduzidos')

plt.subplot(1, 3, 3)
plt.imshow(imagem_preto_branco, cmap='gray')
plt.title('Imagem Preto e Branco')

plt.show()

